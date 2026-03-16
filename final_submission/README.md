# AWS Integrated CTA Bus Data Collection

A Jupyter notebook that continuously polls the Chicago Transit Authority (CTA) Bus Tracker API for real-time vehicle positions across all active routes, writing results to CSV and uploading them to AWS S3 in configurable time-chunked files.

## Overview

This notebook authenticates with AWS, then runs a polling loop that queries the CTA BusTime API every ~30 seconds, collecting GPS position data for every active bus across all 124 routes. Data is accumulated locally into a single CSV and uploaded to S3 at configurable chunk boundaries (default: every 6 hours). On S3 upload success, the local file is deleted; on failure, collection continues uninterrupted into the same file.

## Requirements

### Python Dependencies
- `boto3` — AWS SDK for S3 uploads
- `requests` — HTTP client for CTA API calls
- `pandas` — DataFrame construction and CSV writing
- `zoneinfo` — Timezone handling (Chicago/CST)
- `tempfile`, `pathlib`, `os`, `time`, `argparse` — standard library

### Infrastructure
- **AWS account** with S3 write access to the target bucket
- **CTA BusTime API key** — obtainable from the [CTA developer portal](https://www.ctabustracker.com/home)
- AWS CLI configured and authenticated (`aws login` for SSO-based auth)

## Configuration

The `main()` function accepts the following parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | required | CTA BusTime API key |
| `runtime_seconds` | required | Total duration to run (e.g. `39600` = 11 hours) |
| `per_chunk_sleep` | `5` | Seconds to sleep between route-chunk API calls |
| `per_sweep_sleep` | `30` | Seconds to sleep between full sweeps of all routes |
| `out_dir` | `"data"` | Local directory for the working CSV file |
| `chunk_hours` | `6.0` | Hours per S3 file chunk |
| `s3_bucket` | `"bustrust"` | Target S3 bucket name |
| `no_s3_upload` | `False` | Set to `True` to disable S3 uploads (local-only mode) |

## Usage

1. **Authenticate with AWS:**
```bash
   aws login
```

2. **Run the notebook cell** with your desired parameters:
```python
   main(
       api_key='YOUR_CTA_API_KEY',
       runtime_seconds=39600,
       per_chunk_sleep=5,
       per_sweep_sleep=30,
       out_dir="data/output",
       chunk_hours=6,
       s3_bucket='your-bucket-name',
       no_s3_upload=False,
   )
```

## Output

### Local
A single rolling CSV file: `bus_data_current_chicago.csv`

### S3
Time-stamped files uploaded to `s3://<bucket>/data_collection/` in the format:
```
bus_data_<start_timestamp>_to_<end_timestamp>_chicago.csv
```

### CSV Schema
Each row represents a single vehicle observation at a point in time. Columns mirror the CTA BusTime `getvehicles` API response, plus:
- `pulled_at` — Timestamp when the API call was made (CST)
- `rt_chunk` — The comma-separated route IDs included in that API call

## Data Collection Behavior

- **Routes** are fetched once at startup via `getroutes`, then split into chunks of 10 (the API maximum per call).
- **Sweeps** iterate through all route chunks sequentially, sleeping `per_chunk_sleep` seconds between each call.
- **Chunk rollover** occurs at fixed `chunk_hours` intervals relative to start time. The local file is only uploaded and deleted on success — upload failures do not advance the chunk window, preventing data from spanning incorrect time boundaries.
- **Shutdown upload**: Any remaining data is uploaded on exit, regardless of whether a chunk boundary was reached.

## Known Issues / Notes

- AWS SSO credentials expire after several hours. If the session expires mid-run (as seen in this notebook's output around the 6-hour mark), S3 uploads will begin failing silently while local collection continues. Re-authenticate and restart to resume S3 uploads.
- The CTA API occasionally times out on individual route-chunk calls; these are caught and logged without stopping the sweep.
