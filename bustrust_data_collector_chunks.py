# bustrust_data_collector
"""
pulls CTA Bus Tracker data
writes CSV output in time-chunked files (default: 6-hour chunks)
uploads exactly ONE time per chunk (on rollover), plus one final upload at shutdown
to: s3://<bucket>/data_collection/<filename>

Notes:
- Each API call appends rows to the *current* chunk file.
- When the chunk boundary is crossed, the script:
    1) uploads the completed chunk file to S3 once
    2) rolls over to a new file
"""

import os
import requests
import pandas as pd
import time
import argparse
from datetime import datetime, timedelta
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


def _make_chunk_outfile(out_dir: str, chunk_start: datetime, chunk_hours: float) -> str:
    chunk_end = chunk_start + timedelta(hours=chunk_hours)
    start_stamp = chunk_start.strftime("%Y-%m-%d_%H-%M-%S")
    end_stamp = chunk_end.strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(out_dir, f"bus_data_{start_stamp}_to_{end_stamp}_chicago.csv")


def _upload_file_to_s3(s3, bucket_name: str, local_path: str):
    """
    Upload to s3://bucket/data_collection/<filename>
    """
    filename = os.path.basename(local_path)
    s3_key = os.path.join("data_collection", filename)
    s3.upload_file(local_path, bucket_name, s3_key)
    return s3_key


def main(
    api_key,
    per_chunk_sleep=5,
    per_sweep_sleep=30,
    runtime_seconds=3600,
    out_dir=".",
    chunk_hours=6,
    s3=None,
    s3_bucket="bustrust",
    no_s3_upload=False,
):
    routes = get_routes(api_key)
    chunks = chunk_list(routes, n=10)

    os.makedirs(out_dir, exist_ok=True)

    start = time.time()
    end = start + runtime_seconds

    # time chunking
    if chunk_hours <= 0:
        raise ValueError("chunk_hours must be a positive number of hours (e.g., 6).")
    chunk_seconds = int(chunk_hours * 3600)

    # start first chunk at *script start time* (not aligned to clock boundaries)
    chunk_start_dt = datetime.now(CHICAGO_TZ)
    next_rollover_ts = start + chunk_seconds

    outfiles = []
    outfile = _make_chunk_outfile(out_dir, chunk_start_dt, chunk_hours)
    outfiles.append(outfile)

    print(f"Found {len(routes)} routes -> {len(chunks)} chunks")
    print(f"Writing in {chunk_hours}-hour files under: {os.path.abspath(out_dir)}")
    print(f"Current chunk file: {outfile}")
    print(f"Chunk sleep: {per_chunk_sleep}s | Sweep sleep: {per_sweep_sleep}s | Runtime: {runtime_seconds}s")
    if no_s3_upload:
        print("S3 upload: DISABLED (--no_s3_upload set)")
    else:
        print(f"S3 upload: ENABLED -> s3://{s3_bucket}/data_collection/")

    sweep_num = 0
    call_num = 0
    total_rows = 0

    def upload_current_file(reason: str):
        if no_s3_upload or s3 is None:
            return
        if not os.path.exists(outfile):
            return
        try:
            s3_key = _upload_file_to_s3(s3, s3_bucket, outfile)
            print(f"  [S3] uploaded ({reason}) -> s3://{s3_bucket}/{s3_key}")
        except Exception as e:
            print(f"  [S3] upload ERROR ({reason}): {e}")

    while time.time() < end:
        # roll file if we crossed the chunk boundary
        now_ts = time.time()
        if now_ts >= next_rollover_ts:
            # upload the completed chunk ONCE, then roll over
            upload_current_file("rollover")

            chunk_start_dt = datetime.now(CHICAGO_TZ)
            outfile = _make_chunk_outfile(out_dir, chunk_start_dt, chunk_hours)
            outfiles.append(outfile)
            next_rollover_ts = now_ts + chunk_seconds
            print(f"--- Rolled over to new chunk file: {outfile}")

        sweep_num += 1
        print(f"--- Sweep {sweep_num} @ {datetime.now(CHICAGO_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

        for i, chunk in enumerate(chunks):
            if time.time() >= end:
                break

            # check rollover inside the sweep too (in case a sweep spans the boundary)
            now_ts = time.time()
            if now_ts >= next_rollover_ts:
                upload_current_file("rollover")

                chunk_start_dt = datetime.now(CHICAGO_TZ)
                outfile = _make_chunk_outfile(out_dir, chunk_start_dt, chunk_hours)
                outfiles.append(outfile)
                next_rollover_ts = now_ts + chunk_seconds
                print(f"--- Rolled over to new chunk file: {outfile}")

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
                print(
                    f"[Call {call_num}] chunk {i+1}/{len(chunks)}: appended {n_rows} rows "
                    f"(total {total_rows}) -> {os.path.basename(outfile)}"
                )
            except Exception as e:
                call_num += 1
                print(f"[Call {call_num}] chunk {i+1}/{len(chunks)} ERROR: {e}")

            if time.time() < end:
                time.sleep(min(per_chunk_sleep, max(0, end - time.time())))

        if time.time() < end:
            sleep_now = min(per_sweep_sleep, max(0, end - time.time()))
            print(f"--- Sweep {sweep_num} complete. Sleeping {sleep_now:.0f}s ---")
            time.sleep(sleep_now)

    # final upload of the last (possibly partial) chunk ONCE
    upload_current_file("final")

    print(f" Done. Sweeps: {sweep_num}, calls: {call_num}, total rows written: {total_rows}")
    print("Output files:")
    for f in outfiles:
        print(f"  - {f}")
    return outfiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTA Bus Data Collector")

    parser.add_argument("--api_key", required=True, help="CTA API key")

    parser.add_argument(
        "--runtime_hours",
        type=float,
        default=1.0,
        help="Total runtime in hours (default: 1.0)",
    )

    parser.add_argument(
        "--chunk_sleep",
        type=int,
        default=5,
        help="Sleep between route chunks (default: 5)",
    )
    parser.add_argument(
        "--sweep_sleep",
        type=int,
        default=30,
        help="Sleep between sweeps (default: 30)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--chunk_hours",
        type=float,
        default=6,
        help="Hours per output file (default: 6).",
    )
    parser.add_argument(
        "--s3_bucket",
        type=str,
        default="bustrust",
        help="S3 bucket name (default: bustrust).",
    )
    parser.add_argument(
        "--no_s3_upload",
        action="store_true",
        help="If set, do not upload output files to S3.",
    )

    args = parser.parse_args()

    s3_client = None
    if not args.no_s3_upload:
        s3_client = boto3.client("s3")

    runtime_seconds = int(args.runtime_hours * 3600)
    print(f"Runtime: {args.runtime_hours} hours ({runtime_seconds} seconds)")

    outfiles = main(
        api_key=args.api_key,
        runtime_seconds=runtime_seconds,
        per_chunk_sleep=args.chunk_sleep,
        per_sweep_sleep=args.sweep_sleep,
        out_dir=args.out_dir,
        chunk_hours=args.chunk_hours,
        s3=s3_client,
        s3_bucket=args.s3_bucket,
        no_s3_upload=args.no_s3_upload,
    )


"""
example usage:

python bustrust_data_collector.py \
  --api_key "YOUR_KEY" \
  --runtime_hours 24 \
  --out_dir data \
  --chunk_hours 6 \
  --s3_bucket bustrust

"""