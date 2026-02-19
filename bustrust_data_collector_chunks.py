# chunk2.py 
"""
CTA Bus Tracker data collector 

- Writes to onelocal file at all times:
    <out_dir>/bus_data_current_chicago.csv
- Upload happens at the chunk boundary (chunk_hours).
- On successful upload, the local file is deleted (so it never accumulates multiple chunk files).
- If the upload fails, it keeps writing into the same single local file
- Uploads any data remaining at the finish

S3 destination:
  s3://<bucket>/data_collection/bus_data_<start>_to_<end>_chicago.csv

"""


import os
import time
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import boto3

CHICAGO_TZ = ZoneInfo("America/Chicago")


def chunk_list(xs, n=10):
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def get_routes(session: requests.Session, api_key: str):
    url = f"https://www.ctabustracker.com/bustime/api/v3/getroutes?key={api_key}&format=json"
    r = session.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    routes = data.get("bustime-response", {}).get("routes", [])
    if not routes:
        err = data.get("bustime-response", {}).get("error", [])
        raise ValueError(f"No routes returned. Error: {err}")

    return [rt["rt"] for rt in routes if "rt" in rt]


def get_api(session: requests.Session, url: str):
    r = session.get(url, timeout=30)
    if r.ok:
        return r.json()
    raise ValueError(f"API request failed (status={r.status_code})")


def append_vehicles_to_csv(data, outfile: str, pulled_at: str, rt_chunk: str) -> int:
    vehicles = data.get("bustime-response", {}).get("vehicle", None)
    if not vehicles:
        return 0

    df = pd.DataFrame(vehicles)
    df["pulled_at"] = pulled_at
    df["rt_chunk"] = rt_chunk

    file_exists = os.path.exists(outfile)
    df.to_csv(outfile, mode="a", header=not file_exists, index=False)
    return len(df)


def _current_outfile(out_dir: str) -> str:
    return os.path.join(out_dir, "bus_data_current_chicago.csv")


def _s3_key_for_chunk(chunk_start_dt: datetime, chunk_hours: float) -> str:
    chunk_end_dt = chunk_start_dt + timedelta(hours=chunk_hours)
    start_stamp = chunk_start_dt.strftime("%Y-%m-%d_%H-%M-%S")
    end_stamp = chunk_end_dt.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"bus_data_{start_stamp}_to_{end_stamp}_chicago.csv"
    return os.path.join("data_collection", filename)


def _upload_file_to_s3_with_key(s3, bucket_name: str, local_path: str, s3_key: str) -> str:
    s3.upload_file(local_path, bucket_name, s3_key)
    return s3_key


def main(
    api_key: str,
    runtime_seconds: int,
    per_chunk_sleep: int = 5,
    per_sweep_sleep: int = 30,
    out_dir: str = "data",
    chunk_hours: float = 6.0,
    s3_bucket: str = "bustrust",
    no_s3_upload: bool = False,
):
    if chunk_hours <= 0:
        raise ValueError("chunk_hours must be > 0 (e.g., 6).")
    if runtime_seconds <= 0:
        raise ValueError("runtime_seconds must be > 0.")

    os.makedirs(out_dir, exist_ok=True)

    session = requests.Session()
    routes = get_routes(session, api_key)
    route_chunks = chunk_list(routes, n=10)

    start_ts = time.time()
    end_ts = start_ts + runtime_seconds

    chunk_seconds = int(chunk_hours * 3600)

    # Chunk schedule (fixed relative to start)
    chunk_start_ts = start_ts
    chunk_start_dt = datetime.now(CHICAGO_TZ)  # used for S3 filenames
    next_rollover_ts = chunk_start_ts + chunk_seconds

    # Single local file
    outfile = _current_outfile(out_dir)

    s3_client = None
    if not no_s3_upload:
        s3_client = boto3.client("s3")

    print(f"Found {len(routes)} routes -> {len(route_chunks)} route-chunks")
    print(f"Local output (single file): {os.path.abspath(outfile)}")
    print(f"Chunk: {chunk_hours} hours ({chunk_seconds} seconds)")
    print(f"Sleep: per_chunk={per_chunk_sleep}s | per_sweep={per_sweep_sleep}s | runtime={runtime_seconds}s")
    print("S3 upload:", "DISABLED" if no_s3_upload else f"ENABLED -> s3://{s3_bucket}/data_collection/")
    print("Upload policy: ONLY at chunk boundary; delete local file after successful upload.")
    print("If upload fails: keep writing into the SAME local file; do NOT advance chunk window.")

    sweep_num = 0
    call_num = 0
    total_rows = 0

    def rollover_if_needed():
        """
        At chunk boundary, single local file is uploaded to S3 with a timestamped key, then deleted locally.
        If upload fails, it keeps writing into the same local file and tries again at the next chunk boundary.
        """
        nonlocal chunk_start_ts, chunk_start_dt, next_rollover_ts

        now_ts = time.time()
        if now_ts < next_rollover_ts:
            return

        # If there's nothing written yet, still advance the window (optional behavior).
        # Here, we only try to upload if the file exists and has non-zero size.
        has_data = os.path.exists(outfile) and os.path.getsize(outfile) > 0

        if not no_s3_upload and s3_client is not None and has_data:
            s3_key = _s3_key_for_chunk(chunk_start_dt, chunk_hours)
            try:
                _upload_file_to_s3_with_key(s3_client, s3_bucket, outfile, s3_key)
                print(f"  [S3] uploaded -> s3://{s3_bucket}/{s3_key}")

                # Delete local file so we never keep multiple files.
                os.remove(outfile)
                print(f"  [LOCAL] deleted -> {outfile}")

            except Exception as e:
                print(f"  [S3] upload ERROR (continuing with same local file): {e}")
                # Do NOT advance chunk window if upload failed.
                return

        chunk_start_ts = next_rollover_ts
        chunk_start_dt = chunk_start_dt + timedelta(hours=chunk_hours)
        next_rollover_ts = chunk_start_ts + chunk_seconds
        print(f"--- Rolled over chunk window. Continuing in same local file: {os.path.basename(outfile)}")

    while time.time() < end_ts:
        rollover_if_needed()

        sweep_num += 1
        now_label = datetime.now(CHICAGO_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"--- Sweep {sweep_num} @ {now_label} ---")

        for i, chunk in enumerate(route_chunks):
            if time.time() >= end_ts:
                break

            rollover_if_needed()

            rt_param = ",".join(chunk)
            url = (
                "https://www.ctabustracker.com/bustime/api/v3/getvehicles"
                f"?key={api_key}&rt={rt_param}&format=json"
            )
            pulled_at = datetime.now(CHICAGO_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")

            try:
                data = get_api(session, url)
                n_rows = append_vehicles_to_csv(data, outfile, pulled_at=pulled_at, rt_chunk=rt_param)
                call_num += 1
                total_rows += n_rows
                print(
                    f"[Call {call_num}] routeset {i+1}/{len(route_chunks)}: +{n_rows} rows "
                    f"(total {total_rows}) -> {os.path.basename(outfile)}"
                )
            except Exception as e:
                call_num += 1
                print(f"[Call {call_num}] routeset {i+1}/{len(route_chunks)} ERROR: {e}")

            if time.time() < end_ts:
                time.sleep(min(per_chunk_sleep, max(0, end_ts - time.time())))

        if time.time() < end_ts:
            sleep_now = min(per_sweep_sleep, max(0, end_ts - time.time()))
            print(f"--- Sweep {sweep_num} complete. Sleeping {sleep_now:.0f}s ---")
            time.sleep(sleep_now)

    # ALWAYS upload whatever remains ONCE at shutdown (unless S3 disabled)
    if (not no_s3_upload) and (s3_client is not None):
        if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
            s3_key = _s3_key_for_chunk(chunk_start_dt, chunk_hours)
            try:
                _upload_file_to_s3_with_key(s3_client, s3_bucket, outfile, s3_key)
                print(f"  [S3] uploaded (exit) -> s3://{s3_bucket}/{s3_key}")
                os.remove(outfile)
                print(f"  [LOCAL] deleted (exit) -> {outfile}")
            except Exception as e:
                print(f"  [S3] upload ERROR on exit (file kept): {e}")

    print(f"\nDone. Sweeps: {sweep_num}, calls: {call_num}, total rows written: {total_rows}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTA Bus Data Collector (single local file, S3 uploads at intervals)")

    parser.add_argument("--api_key", required=True, help="CTA API key")

    parser.add_argument("--runtime_hours", type=float, default=1.0, help="Total runtime in hours (default: 1.0)")
    parser.add_argument("--chunk_hours", type=float, default=6.0, help="Hours per chunk boundary (default: 6.0)")

    parser.add_argument("--chunk_sleep", type=int, default=5, help="Sleep between route chunks (default: 5)")
    parser.add_argument("--sweep_sleep", type=int, default=30, help="Sleep between sweeps (default: 30)")

    parser.add_argument("--out_dir", type=str, default="data", help="Output directory (default: data)")

    parser.add_argument("--s3_bucket", type=str, default="bustrust", help="S3 bucket name (default: bustrust)")
    parser.add_argument("--no_s3_upload", action="store_true", help="Disable S3 uploads (still writes locally)")

    args = parser.parse_args()
    runtime_seconds = int(args.runtime_hours * 3600)

    print(f"Runtime: {args.runtime_hours} hours ({runtime_seconds} seconds)")
    main(
        api_key=args.api_key,
        runtime_seconds=runtime_seconds,
        per_chunk_sleep=args.chunk_sleep,
        per_sweep_sleep=args.sweep_sleep,
        out_dir=args.out_dir,
        chunk_hours=args.chunk_hours,
        s3_bucket=args.s3_bucket,
        no_s3_upload=args.no_s3_upload,
    )

"""
Example: 10 minutes total, 3-minute chunks
  runtime_hours = 10/60 = 0.1667
  chunk_hours   = 3/60  = 0.05

python bustrust_data_collector.py \
  --api_key "KEY" \
  --runtime_hours 0.1667 \
  --chunk_hours 0.05 \
  --out_dir data \
  --s3_bucket bustrust
"""