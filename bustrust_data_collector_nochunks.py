
# bustrust_data_collector
'''
pulls data
creates file
pulls more data
loads in file
adds to file
pushes file
repeat until done

'''

import os
import requests
import pandas as pd
import time
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import boto3

CHICAGO_TZ = ZoneInfo("America/Chicago")

def chunk_list(xs, n=10):
    return [xs[i:i+n] for i in range(0, len(xs), n)]

def get_routes(api_key):
    url = f"https://www.ctabustracker.com/bustime/api/v3/getroutes?key={api_key}&format=json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    routes = data.get("bustime-response", {}).get("routes", [])
    if not routes:
        err = data.get("bustime-response", {}).get("error", [])
        raise ValueError(f"No routes returned. Error: {err}")

    return [rt["rt"] for rt in routes if "rt" in rt]

def get_api(url):
    r = requests.get(url, timeout=30)
    if r.ok:
        return r.json()
    raise ValueError(f"API request failed (status={r.status_code})")

def append_vehicles_to_csv(data, outfile, pulled_at, rt_chunk):
    vehicles = data.get("bustime-response", {}).get("vehicle", None)
    if not vehicles:
        # no vehicles is normal sometimes; don't crash the whole run
        return 0

    df = pd.DataFrame(vehicles)

    # add metadata columns so you can trace pulls later
    df["pulled_at"] = pulled_at
    df["rt_chunk"] = rt_chunk

    file_exists = os.path.exists(outfile)
    df.to_csv(outfile, mode="a", header=not file_exists, index=False)
    return len(df)

def main(api_key, per_chunk_sleep=5, per_sweep_sleep=30, runtime_seconds=3600, out_dir="."):
    routes = get_routes(api_key)
    chunks = chunk_list(routes, n=10)

    start_stamp = datetime.now(CHICAGO_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    outfile = os.path.join(out_dir, f"bus_data_{start_stamp}_chicago.csv")

    print(f"Found {len(routes)} routes -> {len(chunks)} chunks")
    print(f"Writing EVERYTHING to one file:\n  {outfile}\n")
    print(f"Chunk sleep: {per_chunk_sleep}s | Sweep sleep: {per_sweep_sleep}s | Runtime: {runtime_seconds}s\n")

    start = time.time()
    end = start + runtime_seconds

    sweep_num = 0
    call_num = 0
    total_rows = 0

    while time.time() < end:
        sweep_num += 1
        print(f"--- Sweep {sweep_num} @ {datetime.now(CHICAGO_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

        for i, chunk in enumerate(chunks):
            if time.time() >= end:
                break

            rt_param = ",".join(chunk)
            url = (
                "https://www.ctabustracker.com/bustime/api/v3/getvehicles"
                f"?key={api_key}&rt={rt_param}&format=json"
            )

            pulled_at = datetime.now(CHICAGO_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")

            try:
                data = get_api(url)
                n_rows = append_vehicles_to_csv(data, outfile, pulled_at=pulled_at, rt_chunk=rt_param)
                call_num += 1
                total_rows += n_rows
                print(f"[Call {call_num}] chunk {i+1}/{len(chunks)}: appended {n_rows} rows (total {total_rows})")
            except Exception as e:
                call_num += 1
                print(f"[Call {call_num}] chunk {i+1}/{len(chunks)} ERROR: {e}")

            if time.time() < end:
                time.sleep(min(per_chunk_sleep, max(0, end - time.time())))

        if time.time() < end:
            sleep_now = min(per_sweep_sleep, max(0, end - time.time()))
            print(f"--- Sweep {sweep_num} complete. Sleeping {sleep_now:.0f}s ---\n")
            time.sleep(sleep_now)

    print(f"\nDone. Sweeps: {sweep_num}, calls: {call_num}, total rows written: {total_rows}")
    print(f"Output file: {outfile}")
    return outfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTA Bus Data Collector")

    parser.add_argument("--api_key", required=True, help="CTA API key")
    parser.add_argument("--runtime", type=int, default=3600,
                        help="Total runtime in seconds (default: 3600)")
    parser.add_argument("--chunk_sleep", type=int, default=5,
                        help="Sleep between route chunks (default: 5)")
    parser.add_argument("--sweep_sleep", type=int, default=30,
                        help="Sleep between sweeps (default: 30)")
    parser.add_argument("--out_dir", type=str, default="data",
                        help="Output directory (default: data)")

    args = parser.parse_args()

    outfile = main(
        api_key=args.api_key,
        runtime_seconds=args.runtime,
        per_chunk_sleep=args.chunk_sleep,
        per_sweep_sleep=args.sweep_sleep,
        out_dir=args.out_dir
    )

    s3 = boto3.client("s3")
    bucket_name = "bustrust"
    s3_key = os.path.basename(outfile)  # keep existing timestamped filename

    try:
        s3.upload_file(outfile, bucket_name, s3_key)
        print(f"Uploaded {outfile} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading {outfile} to S3: {e}")