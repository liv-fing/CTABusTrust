"""
CTA Bus On-Time Analysis — Optimized for Large Data Volumes
============================================================
Designed to handle many gigabytes of vehicle snapshot data efficiently.

Performance strategy
--------------------
1. GTFS pre-processing (run once):
   stop_times.txt → stop_times.parquet  (~10x faster loads, ~5x smaller on disk)
   This is the most important step. Run --preprocess-gtfs once per GTFS release.

2. Vectorized trip matching (replaces row-by-row Python loop):
   For each route, we build numpy arrays of stop distances and scheduled times
   across all trips, then use np.searchsorted and broadcasting to score every
   (vehicle, trip) pair simultaneously. No Python loops over rows.

3. DuckDB for vehicle data:
   Instead of loading all your vehicle CSVs/Parquets into memory at once,
   DuckDB queries them directly from disk using SQL. You can point it at a
   folder of Parquet files covering months of data and it will only read
   what it needs. No cluster required.

4. Parquet output:
   Results are written as Parquet, partitioned by date and route, so
   downstream queries (e.g. "all route 66 data for March") skip irrelevant
   files entirely.

On-time window (CTA standard)
------------------------------
  early   : delay < -1 min
  on_time : -1 min <= delay <= +5 min
  late    : delay > +5 min

Ghost bus detection
-------------------
  G1 - Frozen pdist across polls (requires both G1+G2 to flag)
  G2 - Impossible distance jump between polls (requires both G1+G2 to flag)
  Note: dly is used only as an accuracy checker, not a ghost signal.
  Note: coord mismatch is retained as a diagnostic column (dist_from_next_stop_km)
        but is NOT used as a ghost signal — a bus mid-segment on a long route can
        legitimately be many km from its next stop.

Usage
-----
  # Step 1: Pre-process GTFS once per feed update
  python bus_ontime_analysis.py --preprocess-gtfs \\
      --stop_times stop_times.txt --trips trips.txt --stops stops.txt \\
      --gtfs-cache ./gtfs_cache

  # Step 2: Run analysis (CSV, single Parquet, or directory of files)
  python bus_ontime_analysis.py \\
      --vehicles ./vehicles_data/ \\
      --gtfs-cache ./gtfs_cache \\
      --output ./results/

  # Original single-file CSV usage still works
  python bus_ontime_analysis.py --vehicles vehicles.csv
"""

import argparse
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

EARLY_THRESHOLD_MINUTES = -1   # more than 1 min early  -> early
LATE_THRESHOLD_MINUTES  =  5   # more than 5 min late   -> late

GHOST_FROZEN_PDIST_THRESHOLD = 50     # feet; pdist change below this = frozen
GHOST_FROZEN_MIN_MINUTES     = 20     # minutes; freeze must last this long to flag
GHOST_MIN_OBSERVATIONS       = 2      # need at least this many snapshots per vid
GHOST_MAX_SPEED_FT_PER_MIN  = 7920   # 90 mph in feet per minute
GHOST_DEPARTURE_THRESHOLD    = 1000   # feet; bus never advanced this far = never departed

# Vehicles whose pdist std across all observations is below this threshold
# have a broken pdist sensor and will use time-only matching instead.
PDIST_RELIABLE_STD_THRESHOLD = 100    # feet

# Matches with a time score above this threshold indicate the vehicle is
# operating outside any scheduled GTFS trip window (e.g. unscheduled service,
# extra runs, owl service not in the static feed). These are classified as
# "unscheduled" rather than late/early to avoid false positives.
MAX_MATCH_SCORE_MINUTES = 5.0

FILTER_ROUTES  = True    # only load GTFS trips for routes present in vehicle data
MATCH_BATCH_SIZE = 2_000  # vehicles per matching batch — tune to available RAM


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_gtfs_time(time_str):
    """GTFS time string -> minutes since midnight. Handles times past midnight (>24h)."""
    try:
        h, m, s = map(int, str(time_str).split(":"))
        return h * 60 + m + s / 60
    except Exception:
        return np.nan


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def vehicle_time_to_minutes(tmstmp_series):
    dt = pd.to_datetime(tmstmp_series)
    return dt.dt.hour * 60 + dt.dt.minute + dt.dt.second / 60


def _t(label, t0):
    print(f"    done ({time.time() - t0:.1f}s) — {label}")


# ── Step 0: GTFS pre-processing (run once per feed) ───────────────────────────

def preprocess_gtfs(stop_times_path, trips_path, stops_path, cache_dir,
                    calendar_path=None):
    """
    Merge stop_times + trips + stops into one enriched Parquet file.
    Also saves a trips_calendar.parquet lookup for day-of-week filtering.
    Run once when you get a new GTFS feed.

    Outputs:
        {cache_dir}/stop_times_enriched.parquet
        {cache_dir}/trips_calendar.parquet
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path  = cache_dir / "stop_times_enriched.parquet"
    cal_path  = cache_dir / "trips_calendar.parquet"

    print(f"\n[GTFS PRE-PROCESS] Building {out_path} ...")
    t0 = time.time()

    trips = pd.read_csv(trips_path,
                        dtype={"route_id": str, "trip_id": str,
                               "schd_trip_id": str, "shape_id": str,
                               "service_id": str})
    stops = pd.read_csv(stops_path, dtype={"stop_id": str})
    print(f"    trips: {len(trips):,} rows | stops: {len(stops):,} rows")

    # ── Build trips_calendar lookup ───────────────────────────────────────
    # Maps trip_id → shape_id, route_id, service_id, and day-of-week flags
    # so we can filter candidates by the observation date at match time.
    if calendar_path and Path(calendar_path).exists():
        calendar = pd.read_csv(calendar_path, dtype=str)
        day_cols  = ["monday","tuesday","wednesday","thursday",
                     "friday","saturday","sunday"]
        for col in day_cols:
            calendar[col] = calendar[col].astype(int)
        trips_cal = trips[["trip_id","route_id","shape_id","service_id"]].merge(
            calendar[["service_id"] + day_cols], on="service_id", how="left"
        )
    else:
        print("    ⚠ No calendar.txt provided — day-of-week filtering disabled")
        trips_cal = trips[["trip_id","route_id","shape_id","service_id"]].copy()
        for col in ["monday","tuesday","wednesday","thursday",
                    "friday","saturday","sunday"]:
            trips_cal[col] = 1  # treat all trips as always running

    pq.write_table(
        pa.Table.from_pandas(trips_cal, preserve_index=False),
        cal_path, compression="snappy"
    )
    print(f"    trips_calendar saved: {len(trips_cal):,} rows → {cal_path}")

    # ── Build enriched stop_times ─────────────────────────────────────────
    trip_id_set = set(trips["trip_id"].astype(str))
    chunks = []
    print("    Loading stop_times.txt in chunks...")
    for chunk in pd.read_csv(
        stop_times_path,
        dtype={"trip_id": str, "stop_id": str},
        chunksize=500_000,
    ):
        chunks.append(chunk[chunk["trip_id"].isin(trip_id_set)])
    st = pd.concat(chunks, ignore_index=True)
    print(f"    stop_times rows: {len(st):,}")

    st["arrival_minutes"]     = st["arrival_time"].apply(parse_gtfs_time)
    st["departure_minutes"]   = st["departure_time"].apply(parse_gtfs_time)
    st["shape_dist_traveled"] = pd.to_numeric(st["shape_dist_traveled"], errors="coerce")
    st = st.dropna(subset=["shape_dist_traveled", "arrival_minutes"])
    st = st.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)

    st = st.merge(stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
                  on="stop_id", how="left")
    # Include shape_id in the enriched table so the matcher can use it
    st = st.merge(trips[["trip_id", "route_id", "shape_id",
                          "direction", "schd_trip_id", "service_id"]],
                  on="trip_id", how="left")
    st["route_id"] = st["route_id"].astype(str).str.strip()

    keep = ["trip_id", "stop_id", "stop_sequence", "arrival_minutes",
            "departure_minutes", "shape_dist_traveled", "stop_name",
            "stop_lat", "stop_lon", "route_id", "shape_id",
            "service_id", "direction", "schd_trip_id"]
    st = st[[c for c in keep if c in st.columns]]
    st = st.sort_values(["route_id", "shape_id", "trip_id", "stop_sequence"])

    pq.write_table(
        pa.Table.from_pandas(st, preserve_index=False),
        out_path, row_group_size=100_000, compression="snappy",
    )
    size_mb = out_path.stat().st_size / 1_048_576
    _t(f"Parquet written ({size_mb:.1f} MB) -> {out_path}", t0)
    print(f"[GTFS PRE-PROCESS] Complete. Total: {time.time() - t0:.1f}s\n")
    return str(out_path)


def load_gtfs_cache(cache_dir, route_filter=None):
    """
    Load enriched stop_times and trips_calendar from Parquet cache.
    Returns (stop_times_df, trips_calendar_df).
    """
    cache_dir = Path(cache_dir)
    out_path  = cache_dir / "stop_times_enriched.parquet"
    cal_path  = cache_dir / "trips_calendar.parquet"

    if not out_path.exists():
        raise FileNotFoundError(
            f"GTFS cache not found at {out_path}. "
            "Run with --preprocess-gtfs first."
        )

    t0  = time.time()
    con = duckdb.connect()
    if route_filter:
        routes_sql = ", ".join(f"'{r}'" for r in route_filter)
        st = con.execute(
            f"SELECT * FROM read_parquet('{out_path}') "
            f"WHERE route_id IN ({routes_sql})"
        ).df()
    else:
        st = con.execute(f"SELECT * FROM read_parquet('{out_path}')").df()

    if cal_path.exists():
        trips_cal = con.execute(f"SELECT * FROM read_parquet('{cal_path}')").df()
    else:
        print("    ⚠ trips_calendar.parquet not found — re-run --preprocess-gtfs "
              "with --calendar to enable day-of-week filtering")
        trips_cal = None

    con.close()
    _t(f"GTFS cache loaded: {len(st):,} stop_times", t0)
    return st, trips_cal


# ── Step 1: Load vehicles ──────────────────────────────────────────────────────

def load_vehicles(vehicles_path):
    """
    Load vehicle snapshots via DuckDB. Accepts:
      - A single CSV file
      - A single Parquet file
      - A directory of CSV or Parquet files (all read in one scan)

    DuckDB reads only the columns and row groups it needs, keeping memory
    usage low even when the total data is many gigabytes.
    """
    print(f"[1/6] Loading vehicles from {vehicles_path} ...")
    t0 = time.time()

    p = Path(vehicles_path)
    con = duckdb.connect()

    if p.is_dir():
        parquets = list(p.glob("*.parquet"))
        if parquets:
            df = con.execute(f"SELECT * FROM read_parquet('{p}/*.parquet')").df()
        else:
            df = con.execute(
                f"SELECT * FROM read_csv_auto('{p}/*.csv', all_varchar=true, ignore_errors=true)"
            ).df()
    elif p.suffix == ".parquet":
        df = con.execute(f"SELECT * FROM read_parquet('{p}')").df()
    else:
        df = con.execute(
            f"SELECT * FROM read_csv_auto('{p}', all_varchar=true, ignore_errors=true)"
        ).df()
    con.close()

    # Drop stale columns from any previous merge attempts
    df = df.drop(columns=[c for c in
                           ["trip_id", "schd_trip_id", "route_id", "tatripid_str"]
                           if c in df.columns])

    df["tmstmp"]       = pd.to_datetime(df["tmstmp"], errors="coerce")
    df["dly"]          = (df["dly"].astype(str).str.lower()
                          .map({"true": True, "false": False}).fillna(False))
    df["rt"]           = df["rt"].astype(str).str.strip()
    df["pdist"]        = pd.to_numeric(df["pdist"], errors="coerce")
    df["vid"]          = df["vid"].astype(str).str.strip()
    df["tatripid"]     = pd.to_numeric(df["tatripid"], errors="coerce")  # bad values → NaN
    df["lat"]          = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"]          = pd.to_numeric(df["lon"], errors="coerce")
    df["time_minutes"] = vehicle_time_to_minutes(df["tmstmp"])

    # Report bad rows so data quality issues are visible
    bad_tmstmp = df["tmstmp"].isna().sum()
    bad_pdist  = df["pdist"].isna().sum()
    bad_lat    = df["lat"].isna().sum()
    if any([bad_tmstmp, bad_pdist, bad_lat]):
        print(f"    ⚠ Rows with bad values (set to NaN): "
              f"tmstmp={bad_tmstmp}, pdist={bad_pdist}, lat/lon={bad_lat}")

    # Drop rows with no timestamp or position — they can't be matched
    before_drop = len(df)
    df = df.dropna(subset=["tmstmp", "pdist", "lat", "lon"]).reset_index(drop=True)
    dropped = before_drop - len(df)
    if dropped > 0:
        print(f"    ⚠ Dropped {dropped} rows missing timestamp or position")
    df = df.reset_index(drop=True)

    # Deduplicate — the Bus Tracker API occasionally returns the same vehicle
    # twice in the same poll. Keep the first occurrence.
    before = len(df)
    df = df.drop_duplicates(subset=["vid", "tmstmp"], keep="first").reset_index(drop=True)
    dupes = before - len(df)
    if dupes > 0:
        print(f"    ⚠ Dropped {dupes} duplicate (vid, tmstmp) rows")

    _t(f"{len(df):,} observations | {df['vid'].nunique():,} vehicles "
       f"| {df['rt'].nunique()} routes", t0)
    return df


# ── Step 2: Ghost bus detection ────────────────────────────────────────────────

def detect_ghost_buses(vehicles):
    """
    Ghost bus detection for multi-hour datasets.

    G1 - SUSTAINED FREEZE: pdist unchanged for 20+ consecutive minutes.
    G2 - IMPOSSIBLE FORWARD SPEED: pdist advances faster than 90 mph.
    G3 - NEVER DEPARTED: pdist range across all observations for a (vid, rt)
         pair is below GHOST_DEPARTURE_THRESHOLD. The bus never meaningfully
         left its starting position — almost always a terminal layover ghost
         or a bus that pulled in but never started its run.

    G1 and G3 are independent — G3 fires on the full observation window,
    G1 fires on consecutive-pair intervals. A bus with only 2-3 observations
    all near pdist=0 will be caught by G3 but not G1.
    """
    print("[2/6] Detecting ghost buses...")
    t0 = time.time()
    df = vehicles.copy().sort_values(["vid", "rt", "tmstmp"])

    if df.groupby("vid").size().max() >= GHOST_MIN_OBSERVATIONS:
        g = df.groupby(["vid", "rt"])

        df["_prev_pdist"]     = g["pdist"].shift(1)
        df["_prev_tmstmp"]    = g["tmstmp"].shift(1)
        df["_pdist_delta"]    = df["pdist"] - df["_prev_pdist"]
        df["_time_delta_min"] = (
            (df["tmstmp"] - df["_prev_tmstmp"]).dt.total_seconds() / 60
        )

        df["_prev_was_reset"] = df["_pdist_delta"] < -GHOST_MAX_SPEED_FT_PER_MIN

        # G1: sustained freeze
        df["ghost_frozen"] = (
            (df["_pdist_delta"].abs() < GHOST_FROZEN_PDIST_THRESHOLD)
            & (df["_time_delta_min"] >= GHOST_FROZEN_MIN_MINUTES)
            & df["_prev_pdist"].notna()
        )

        # G2: impossible forward speed
        df["_implied_speed"] = df["_pdist_delta"] / df["_time_delta_min"].clip(lower=0.5)
        df["ghost_jump"] = (
            (df["_implied_speed"] > GHOST_MAX_SPEED_FT_PER_MIN)
            & ~df["_prev_was_reset"].shift(-1).fillna(False)
            & df["_prev_pdist"].notna()
            & (df["_time_delta_min"] > 0)
        )

        df = df.drop(columns=["_prev_pdist", "_prev_tmstmp", "_pdist_delta",
                               "_time_delta_min", "_prev_was_reset", "_implied_speed"])

        # G3: never departed — pdist range across entire (vid, rt) window
        pdist_range = (
            g["pdist"]
            .apply(lambda x: x.max() - x.min())
            .reset_index()
            .rename(columns={"pdist": "_pdist_range"})
        )
        df = df.merge(pdist_range, on=["vid", "rt"], how="left")
        df["ghost_never_departed"] = df["_pdist_range"] < GHOST_DEPARTURE_THRESHOLD
        df = df.drop(columns=["_pdist_range"])

    else:
        print("    ⚠ Single snapshot per vehicle — ghost detection limited")
        df["ghost_frozen"]        = False
        df["ghost_jump"]          = False
        df["ghost_never_departed"] = False

    df["ghost_score"] = (
        df["ghost_frozen"].astype(int)
        + df["ghost_jump"].astype(int)
        + df["ghost_never_departed"].astype(int)
    )

    # G1 or G3 alone are sufficient. G2 alone is not (single bad GPS ping).
    df["is_ghost"] = (
        df["ghost_frozen"]
        | df["ghost_never_departed"]
        | (df["ghost_score"] >= 2)
    )

    n_ghost_vids   = df[df["is_ghost"]]["vid"].nunique()
    n_frozen       = df["ghost_frozen"].sum()
    n_never_dep    = df["ghost_never_departed"].sum()
    n_jump         = df["ghost_jump"].sum()
    _t(
        f"Ghost buses: {n_ghost_vids} vehicles "
        f"(frozen={n_frozen:,}, never_departed={n_never_dep:,}, jump={n_jump:,})",
        t0
    )
    return df


# ── Step 3: Build vectorized trip index ───────────────────────────────────────

def build_candidate_index(stop_times, trips_cal):
    """
    Build two complementary indexes for matching:

    1. candidate_index[shape_id] → list of trip dicts
       Keyed by full shape_id string. Vehicles find their candidates by
       looking up all shape_ids that end with their pid string.

    2. route_index[route_id] → list of trip dicts
       Fallback for vehicles with missing/invalid pid.

    3. pid_to_shapes[pid_str] → [shape_id, ...]
       Pre-built suffix lookup so match_vehicle_to_trip never does string
       scanning at runtime. Built by indexing every numeric suffix of every
       shape_id — pid "7873" hits shape_id "67707873" because it ends with it.

    trips_cal provides day-of-week service flags.
    """
    print("[3/6] Building vectorized trip index...")
    t0 = time.time()

    day_cols = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    svc_days = {}
    if trips_cal is not None and all(c in trips_cal.columns for c in day_cols):
        for _, row in trips_cal.iterrows():
            active = {i for i, col in enumerate(day_cols)
                      if str(row.get(col, "0")) == "1"}
            svc_days[str(row["service_id"])] = active
        trip_service = dict(zip(
            trips_cal["trip_id"].astype(str),
            trips_cal["service_id"].astype(str)
        ))
    else:
        trip_service = {}

    candidate_index = {}   # shape_id → [trip_dict, ...]
    route_index     = {}   # route_id → [trip_dict, ...]
    n_trips = 0

    for (route_id, shape_id), grp in stop_times.groupby(["route_id", "shape_id"]):
        route_id = str(route_id).strip()
        shape_id = str(shape_id).strip()
        trips_for_shape = []

        for trip_id, trip_df in grp.groupby("trip_id"):
            t = trip_df.sort_values("stop_sequence")
            service_id = trip_service.get(str(trip_id), None)
            trip_dict = {
                "trip_id":                trip_id,
                "shape_id":               shape_id,
                "service_id":             service_id,
                "active_days":            svc_days.get(service_id, set(range(7))),
                "dists":                  t["shape_dist_traveled"].values.astype(np.float32),
                "arr_minutes":            t["arrival_minutes"].values.astype(np.float32),
                "dep_minutes":            t["departure_minutes"].values.astype(np.float32),
                "stop_ids":               t["stop_id"].values,
                "stop_names":             t["stop_name"].values,
                "stop_lats":              t["stop_lat"].values.astype(np.float32),
                "stop_lons":              t["stop_lon"].values.astype(np.float32),
                "first_stop_name":        t["stop_name"].values[0],
                "first_stop_dep_minutes": float(t["departure_minutes"].values[0]),
            }
            trips_for_shape.append(trip_dict)
            n_trips += 1

        candidate_index[shape_id] = trips_for_shape

        if route_id not in route_index:
            route_index[route_id] = []
        route_index[route_id].extend(trips_for_shape)

    # Pre-build pid suffix → [shape_id, ...] lookup.
    # Index every numeric suffix of every shape_id so lookup at match time
    # is a single dict get() with no string scanning.
    pid_to_shapes = {}
    for shape_id in candidate_index:
        for length in range(1, len(shape_id) + 1):
            suffix = shape_id[-length:]
            if not suffix.isdigit():
                break  # stop at first non-digit from the right
            if suffix not in pid_to_shapes:
                pid_to_shapes[suffix] = []
            if shape_id not in pid_to_shapes[suffix]:
                pid_to_shapes[suffix].append(shape_id)

    _t(f"{n_trips:,} trips indexed across "
       f"{len(route_index)} routes / {len(candidate_index)} shapes", t0)
    return candidate_index, route_index, pid_to_shapes


def flag_unreliable_pdist(vehicles):
    """
    Detect vehicles whose pdist is static or near-static across many observations
    while their timestamps are progressing — indicating a broken pdist sensor.

    These vehicles cannot be matched by position and will be routed to the
    time-only fallback matcher.

    A vehicle is flagged if its pdist standard deviation across all observations
    is below PDIST_RELIABLE_STD_THRESHOLD feet. A normally moving bus across
    a full trip will have std >> 1000ft.
    """
    pdist_std = (
        vehicles.groupby(["vid", "rt"])["pdist"]
        .std()
        .reset_index()
        .rename(columns={"pdist": "pdist_std"})
        .fillna(0)
    )
    pdist_std["pdist_unreliable"] = pdist_std["pdist_std"] < PDIST_RELIABLE_STD_THRESHOLD

    vehicles = vehicles.merge(pdist_std[["vid", "rt", "pdist_unreliable"]],
                               on=["vid", "rt"], how="left")
    vehicles["pdist_unreliable"] = vehicles["pdist_unreliable"].fillna(False)

    n_unreliable = vehicles[vehicles["pdist_unreliable"]]["vid"].nunique()
    n_obs        = vehicles["pdist_unreliable"].sum()
    if n_unreliable > 0:
        print(f"    ⚠ Unreliable pdist: {n_unreliable} vehicles "
              f"({n_obs:,} observations) — will use time-only matching")
    return vehicles
    """
    Convert the stop_times DataFrame into a dict of numpy arrays, keyed by
    route_id. This is done once up front so the matcher never touches a
    DataFrame or does any pandas operations inside its inner loop.

    Structure:
        trip_index[route_id] = [
            {
                "trip_id":     str,
                "dists":       float32 array  (shape_dist_traveled, sorted),
                "arr_minutes": float32 array  (arrival_minutes),
                "dep_minutes": float32 array  (departure_minutes),
                "stop_ids":    object array,
                "stop_names":  object array,
                "stop_lats":   float32 array,
                "stop_lons":   float32 array,
            },
            ...
        ]
    """
    print("[3/6] Building vectorized trip index...")
    t0 = time.time()
    index = {}
    n_trips = 0

    for route_id, route_df in stop_times.groupby("route_id"):
        trips_for_route = []
        for trip_id, trip_df in route_df.groupby("trip_id"):
            t = trip_df.sort_values("stop_sequence")
            trips_for_route.append({
                "trip_id":     trip_id,
                "dists":       t["shape_dist_traveled"].values.astype(np.float32),
                "arr_minutes": t["arrival_minutes"].values.astype(np.float32),
                "dep_minutes": t["departure_minutes"].values.astype(np.float32),
                "stop_ids":    t["stop_id"].values,
                "stop_names":  t["stop_name"].values,
                "stop_lats":   t["stop_lat"].values.astype(np.float32),
                "stop_lons":   t["stop_lon"].values.astype(np.float32),
            })
            n_trips += 1
        index[str(route_id).strip()] = trips_for_route

    _t(f"{n_trips:,} trips indexed across {len(index)} routes", t0)
    return index


# ── Step 4: Vectorized trip matching ──────────────────────────────────────────

def _match_batch(batch_pdists, batch_times, trips_for_route):
    """
    Core vectorized matching for one batch of vehicles on one route.

    For each trip j, we run np.searchsorted once across ALL vehicles
    simultaneously to find each vehicle's stop bracket, then compute
    the interpolated scheduled time and score in pure numpy.

    Returns:
        best_trip_idx  : int32 array (index into trips_for_route, -1 = no match)
        best_bracket   : int32 array (index of the prev stop in the bracket)
        best_scores    : float32 array (time difference in minutes, lower = better)
    """
    n_veh = len(batch_pdists)
    best_scores   = np.full(n_veh, np.inf, dtype=np.float32)
    best_trip_idx = np.full(n_veh, -1,    dtype=np.int32)
    best_bracket  = np.full(n_veh, -1,    dtype=np.int32)

    for j, trip in enumerate(trips_for_route):
        dists = trip["dists"]   # sorted float32 array of shape_dist_traveled
        dep   = trip["dep_minutes"]
        arr   = trip["arr_minutes"]

        # Find the stop bracket index for every vehicle in one call
        idxs = np.searchsorted(dists, batch_pdists, side="right") - 1

        # Keep only vehicles whose pdist falls inside this trip
        # (not before the first stop or past the last stop)
        valid = (idxs >= 0) & (idxs < len(dists) - 1)
        if not valid.any():
            continue

        v = np.where(valid)[0]   # vehicle positions within the batch
        b = idxs[v]              # corresponding stop bracket indices

        dist_range = dists[b + 1] - dists[b]
        good = dist_range > 0   # guard against duplicate distances
        v, b, dist_range = v[good], b[good], dist_range[good]

        # Linear interpolation of scheduled time at each vehicle's exact pdist
        frac        = (batch_pdists[v] - dists[b]) / dist_range
        interp_time = dep[b] + frac * (arr[b + 1] - dep[b])
        scores      = np.abs(batch_times[v] - interp_time)

        # Update best match where this trip beats the current best
        improve          = scores < best_scores[v]
        improved_v       = v[improve]
        best_scores[improved_v]   = scores[improve]
        best_trip_idx[improved_v] = j
        best_bracket[improved_v]  = b[improve]

    return best_trip_idx, best_bracket, best_scores


def _time_only_match(veh_time, trips_for_route):
    """
    Fallback matcher for vehicles with unreliable pdist.
    Finds the trip whose scheduled time window best contains the vehicle's
    current timestamp, without using position at all.
    """
    best_score   = np.inf
    best_trip    = None
    best_bracket = None

    for trip in trips_for_route:
        arr = trip["arr_minutes"]
        dep = trip["dep_minutes"]

        trip_start = dep[0]
        trip_end   = arr[-1]

        # Skip trips not running near the vehicle's current time (30 min buffer)
        if veh_time < trip_start - 30 or veh_time > trip_end + 30:
            continue

        # Find the stop bracket whose scheduled window contains veh_time
        for i in range(len(arr) - 1):
            if dep[i] <= veh_time <= arr[i + 1]:
                score = abs(veh_time - (dep[i] + arr[i + 1]) / 2)
                if score < best_score:
                    best_score   = score
                    best_trip    = trip
                    best_bracket = i
                break

    return best_trip, best_bracket, best_score


def _build_result(trip, b, frac, vtime, pdist_match_used, score):
    """Build the result dict for a matched vehicle observation."""
    interp_time = (trip["dep_minutes"][b] +
                   frac * (trip["arr_minutes"][b + 1] - trip["dep_minutes"][b]))
    schedule_elapsed = interp_time - trip["first_stop_dep_minutes"]
    return {
        "matched_trip_id":                trip["trip_id"],
        "prev_stop_id":                   trip["stop_ids"][b],
        "prev_stop_name":                 trip["stop_names"][b],
        "prev_stop_dist":                 float(trip["dists"][b]),
        "next_stop_id":                   trip["stop_ids"][b + 1],
        "next_stop_name":                 trip["stop_names"][b + 1],
        "next_stop_dist":                 float(trip["dists"][b + 1]),
        "next_stop_lat":                  float(trip["stop_lats"][b + 1]),
        "next_stop_lon":                  float(trip["stop_lons"][b + 1]),
        "scheduled_arrival_next_minutes": float(trip["arr_minutes"][b + 1]),
        # First-stop anchor fields
        "first_stop_name":                trip["first_stop_name"],
        "scheduled_departure_stop1":      trip["first_stop_dep_minutes"],
        "schedule_elapsed_minutes":       round(float(schedule_elapsed), 1),
        # Match metadata
        "pdist_match_used":               pdist_match_used,
        "delay_minutes":                  round(float(vtime - interp_time), 1),
        "time_match_score_minutes":       round(float(score), 2),
    }


def match_vehicle_to_trip(vehicles, candidate_index, route_index, pid_to_shapes):
    """
    Trip matching using stst (scheduled start time) as the primary key.

    Builds stst_index from actual vehicle pid values at runtime so the
    substring check uses exactly the pid strings vehicles carry.

    Lookup hierarchy per vehicle:
      1. STST (primary): (pid_str, stst_minutes) → exact trip, O(1)
      2. PID+weekday fallback: pid suffix → candidate shape_ids → score-based
      3. Route+weekday fallback: all trips for route → score-based
    """
    print("[4/6] Matching vehicles to scheduled trips (stst-anchored)...")
    t0 = time.time()

    vehicles = vehicles.copy()
    vehicles["_pid_str"]  = vehicles["pid"].astype(str).str.strip()
    vehicles["_stst_min"] = pd.to_numeric(vehicles["stst"], errors="coerce") / 60.0
    vehicles["_weekday"]  = vehicles["tmstmp"].apply(lambda t: pd.Timestamp(t).weekday())

    # Build stst_index from actual vehicle pid values at runtime.
    # pid in the vehicle feed is a number (e.g. 6351) whose string form is
    # always a suffix of the corresponding shape_id (e.g. "67706351").
    # We use endswith — not substring — to avoid false matches from short
    # pids like 95 or 100 appearing inside unrelated shape_ids.
    print("    Building stst_index from vehicle pids...")
    unique_pids = [str(p).strip() for p in vehicles["_pid_str"].dropna().unique()
                   if str(p).strip() not in ("", "nan", "None")]
    stst_index  = {}
    for pid_str in unique_pids:
        for shape_id, trips in candidate_index.items():
            if str(shape_id).endswith(pid_str):
                for trip in trips:
                    key = (pid_str, round(trip["first_stop_dep_minutes"], 1))
                    stst_index[key] = trip
    print(f"    stst_index: {len(stst_index):,} keys for {len(unique_pids):,} unique pids")

    result_rows  = [None] * len(vehicles)
    n_stst_hit   = 0
    n_pid_hit    = 0
    n_route_hit  = 0
    unmatched    = 0

    def _interpolate_and_store(orig_idx, trip, pdist, vtime, unreliable):
        """Given a confirmed trip, find bracket and build result."""
        nonlocal unmatched
        if unreliable:
            t, b, score = _time_only_match(float(vtime), [trip])
            if t is None:
                result_rows[orig_idx] = {}
                unmatched += 1
                return
            result_rows[orig_idx] = _build_result(t, b, 0.5, float(vtime),
                                                   pdist_match_used=False, score=score)
            return

        try:
            pdist_f = float(pdist)
        except (TypeError, ValueError):
            result_rows[orig_idx] = {}
            unmatched += 1
            return
        if np.isnan(pdist_f):
            result_rows[orig_idx] = {}
            unmatched += 1
            return

        best_trip_idx, best_bracket, best_scores = _match_batch(
            np.array([pdist_f], dtype=np.float32),
            np.array([float(vtime)], dtype=np.float32),
            [trip],
        )
        if best_trip_idx[0] == -1:
            result_rows[orig_idx] = {}
            unmatched += 1
            return
        b     = best_bracket[0]
        denom = trip["dists"][b + 1] - trip["dists"][b]
        frac  = float((pdist_f - trip["dists"][b]) / denom) if denom else 0.5
        result_rows[orig_idx] = _build_result(trip, b, frac, float(vtime),
                                               pdist_match_used=True, score=best_scores[0])

    def _score_batch(indices, pdists, times, unreliable_flags, candidates):
        """Score-based fallback: match a batch against a candidate list."""
        nonlocal unmatched
        # Handle unreliable pdist rows individually
        for i, orig_idx in enumerate(indices):
            if unreliable_flags[i]:
                trip, b, score = _time_only_match(float(times[i]), candidates)
                if trip is None:
                    result_rows[orig_idx] = {}
                    unmatched += 1
                else:
                    result_rows[orig_idx] = _build_result(
                        trip, b, 0.5, float(times[i]),
                        pdist_match_used=False, score=score)

        reliable_mask = ~np.array(unreliable_flags, dtype=bool) & ~np.isnan(pdists)
        if not reliable_mask.any():
            return

        r_idx    = [indices[i] for i in range(len(indices)) if reliable_mask[i]]
        r_pdists = pdists[reliable_mask].astype(np.float32)
        r_times  = times[reliable_mask].astype(np.float32)

        for start in range(0, len(r_idx), MATCH_BATCH_SIZE):
            end          = min(start + MATCH_BATCH_SIZE, len(r_idx))
            batch_idx    = r_idx[start:end]
            batch_pdists = r_pdists[start:end]
            batch_times  = r_times[start:end]

            best_trip_idx, best_bracket, best_scores = _match_batch(
                batch_pdists, batch_times, candidates
            )
            for j, orig_idx in enumerate(batch_idx):
                if best_trip_idx[j] == -1:
                    result_rows[orig_idx] = {}
                    unmatched += 1
                    continue
                trip  = candidates[best_trip_idx[j]]
                b     = best_bracket[j]
                denom = trip["dists"][b + 1] - trip["dists"][b]
                frac  = float((batch_pdists[j] - trip["dists"][b]) / denom) if denom else 0.5
                result_rows[orig_idx] = _build_result(
                    trip, b, frac, float(batch_times[j]),
                    pdist_match_used=True, score=best_scores[j])

    # ── Primary pass: group by (pid, stst_min) — exact trip per group ────────
    stst_groups = vehicles.groupby(["_pid_str", "_stst_min"])
    for (pid_str, stst_min), grp in stst_groups:
        if pd.isna(stst_min):
            continue
        stst_key = (str(pid_str), round(float(stst_min), 1))
        trip     = stst_index.get(stst_key)

        if trip is not None:
            n_stst_hit += len(grp)
            for orig_idx, row in grp.iterrows():
                _interpolate_and_store(
                    orig_idx,
                    trip,
                    row["pdist"],
                    row["time_minutes"],
                    row["pdist_unreliable"],
                )
        # Unmatched stst groups fall through to the fallback passes below

    # ── Fallback: rows not yet matched — group by (pid, rt, weekday) ─────────
    unresolved_mask = pd.Series([result_rows[i] is None for i in range(len(vehicles))],
                                index=vehicles.index)
    unresolved = vehicles[unresolved_mask]

    if len(unresolved):
        for (pid_str, rt, obs_wd), grp in unresolved.groupby(["_pid_str", "rt", "_weekday"]):
            rt     = str(rt).strip()
            obs_wd = int(obs_wd)

            candidates = None
            if pid_str and pid_str not in ("nan", "None", ""):
                matching_shapes = pid_to_shapes.get(pid_str, [])
                pid_candidates  = [
                    t for shape_id in matching_shapes
                    for t in candidate_index.get(shape_id, [])
                    if t["shape_id"].endswith(pid_str)
                ]
                if pid_candidates:
                    day_filtered = [t for t in pid_candidates
                                    if obs_wd in t["active_days"]]
                    candidates   = day_filtered if day_filtered else pid_candidates
                    n_pid_hit   += len(grp)

            if not candidates:
                all_rt       = route_index.get(rt, [])
                day_filtered = [t for t in all_rt if obs_wd in t["active_days"]]
                candidates   = day_filtered if day_filtered else all_rt
                n_route_hit += len(grp)

            if not candidates:
                for orig_idx in grp.index:
                    result_rows[orig_idx] = {}
                unmatched += len(grp)
                continue

            _score_batch(
                indices          = grp.index.tolist(),
                pdists           = grp["pdist"].values.astype(float),
                times            = grp["time_minutes"].values.astype(float),
                unreliable_flags = grp["pdist_unreliable"].values.tolist(),
                candidates       = candidates,
            )

    # Fill any remaining Nones
    for i in range(len(result_rows)):
        if result_rows[i] is None:
            result_rows[i] = {}
            unmatched += 1

    vehicles = vehicles.drop(columns=["_pid_str", "_stst_min", "_weekday"])
    matched  = len(vehicles) - unmatched
    _t(f"Matched {matched:,} / {len(vehicles):,} vehicles "
       f"(stst={n_stst_hit:,} pid={n_pid_hit:,} route={n_route_hit:,})", t0)

    match_df = pd.DataFrame(result_rows, index=vehicles.index)
    return pd.concat([vehicles, match_df], axis=1)


# ── Step 5: Compute delay ──────────────────────────────────────────────────────

def compute_delay(df):
    """
    delay_minutes is already computed during matching as:
        vehicle_time - interpolated_scheduled_time_at_current_pdist

    Positive = late. Negative = early.

    "Unscheduled" logic:
    A high match score on its own doesn't mean unscheduled — a genuinely
    late bus will also have a high match score because lateness IS the
    time difference. So we only classify as unscheduled when the match
    score is high AND the delay is small (i.e. the bus appears on time
    but the matcher had low confidence finding any trip for it at all).
    A bus with a high match score AND a high delay is just late.
    """
    print("[5/6] Computing delay...")
    t0 = time.time()

    # Overnight wrap correction
    df.loc[df["delay_minutes"] < -720, "delay_minutes"] += 1440
    df.loc[df["delay_minutes"] >  720, "delay_minutes"] -= 1440

    matched  = df["matched_trip_id"].notna() & df["delay_minutes"].notna()
    is_ghost = df["is_ghost"].astype(bool)
    low_conf = df["time_match_score_minutes"] > MAX_MATCH_SCORE_MINUTES
    d        = df["delay_minutes"]

    df["on_time_status"] = np.select(
        condlist=[
            ~matched,
            is_ghost,
            matched & ~is_ghost & low_conf,
            matched & ~is_ghost & ~low_conf & (d < EARLY_THRESHOLD_MINUTES),
            matched & ~is_ghost & ~low_conf & (d >= EARLY_THRESHOLD_MINUTES) & (d <= LATE_THRESHOLD_MINUTES),
            matched & ~is_ghost & ~low_conf & (d > LATE_THRESHOLD_MINUTES),
        ],
        choicelist=["unmatched", "ghost", "unscheduled", "early", "on_time", "late"],
        default="unmatched",
    )

    # Coord mismatch — diagnostic column only, not a ghost signal
    has_coords = df["next_stop_lat"].notna() & df["lat"].notna()
    df.loc[has_coords, "dist_from_next_stop_km"] = haversine_km(
        df.loc[has_coords, "lat"],   df.loc[has_coords, "lon"],
        df.loc[has_coords, "next_stop_lat"], df.loc[has_coords, "next_stop_lon"],
    ).round(3)

    n_unscheduled = (df["on_time_status"] == "unscheduled").sum()
    if n_unscheduled > 0:
        print(f"    ⚠ {n_unscheduled:,} observations classified as unscheduled "
              f"(match score > {MAX_MATCH_SCORE_MINUTES} min, delay within window)")

    _t("delay computed", t0)
    return df


# ── Step 6: dly accuracy check ────────────────────────────────────────────────

def check_dly_accuracy(df):
    """
    Cross-tab our delay classifications against the API's dly field.
      dly=True  should map to our "late"
      dly=False should map to "on_time" or "early"

    Reports overall agreement, per-route breakdown sorted worst-first,
    and match score distribution so we can see how many observations
    are being discarded by the low-confidence filter.
    """
    # Match score health — how many observations have bad scores?
    scored = df["time_match_score_minutes"].dropna()
    if len(scored):
        print(f"\n  Match score distribution (lower = better fit):")
        for thresh, label in [(1, "<1 min"), (3, "<3 min"), (5, "<5 min"),
                              (10, "<10 min"), (30, "<30 min")]:
            pct = (scored < thresh).mean() * 100
            print(f"    {label:10s}: {pct:.1f}%")
        print(f"    max score  : {scored.max():.1f} min")
        low_conf_n = (scored > MAX_MATCH_SCORE_MINUTES).sum()
        print(f"  Low-confidence matches (score >{MAX_MATCH_SCORE_MINUTES}): "
              f"{low_conf_n:,} ({low_conf_n/len(df)*100:.1f}%)")

    check = df[
        df["on_time_status"].isin(["on_time", "early", "late"])
        & df["dly"].notna()
    ].copy()
    if len(check) == 0:
        print("    ⚠ No matched non-ghost vehicles to check dly accuracy against.")
        return None

    check["dly_says_late"] = check["dly"].astype(bool)
    check["we_say_late"]   = check["on_time_status"] == "late"
    check["agree"]         = check["dly_says_late"] == check["we_say_late"]

    n   = len(check)
    pct = check["agree"].mean() * 100
    fp  = (~check["we_say_late"] &  check["dly_says_late"]).sum()
    fn  = ( check["we_say_late"] & ~check["dly_says_late"]).sum()

    print(f"\n  dly ACCURACY CHECK")
    print(f"  {'─'*44}")
    print(f"  Overall agreement:         {pct:.1f}%  ({check['agree'].sum():,} / {n:,})")
    print(f"  API late, we say ok:       {fp:,}  ({fp/n*100:.1f}%)  <- under-detection")
    print(f"  We say late, API says ok:  {fn:,}  ({fn/n*100:.1f}%)  <- over-detection")

    route_agg = (
        check.groupby("rt")["agree"]
        .agg(n="count", agreement_pct=lambda x: x.mean() * 100)
        .round(1)
        .sort_values("agreement_pct")
    )
    print(f"\n  Routes with lowest agreement (investigate matching here first):")
    print(route_agg.head(10).to_string())
    return route_agg


# ── Step 7: Summarise and save ─────────────────────────────────────────────────

def summarise_and_save(df, output_path):
    print("[6/6] Summarising and saving...")
    t0 = time.time()

    total   = len(df)
    matched = df[df["on_time_status"].isin(["on_time", "early", "late"])]
    counts  = df["on_time_status"].value_counts()

    print(f"\n{'='*52}")
    print(f"  ON-TIME SUMMARY  ({total:,} vehicle observations)")
    print(f"  Window: <{EARLY_THRESHOLD_MINUTES} min = early | "
          f"{EARLY_THRESHOLD_MINUTES} to +{LATE_THRESHOLD_MINUTES} min = on_time | "
          f">{LATE_THRESHOLD_MINUTES} min = late")
    print(f"{'='*52}")
    for status, count in counts.items():
        print(f"  {status:<12}: {count:>6,}  ({count/total*100:.1f}%)")

    if len(matched):
        print(f"\n  Avg delay (matched, non-ghost): {matched['delay_minutes'].mean():.1f} min")
        print(f"  Median delay:                   {matched['delay_minutes'].median():.1f} min")

    ghost_df = df[df["is_ghost"]]
    if len(ghost_df):
        print(f"\n  Ghost buses: {ghost_df['vid'].nunique():,} unique vehicles")
        print(f"    Frozen pdist:    {ghost_df['ghost_frozen'].sum():,}")
        print(f"    Never departed:  {ghost_df['ghost_never_departed'].sum():,}")
        print(f"    Impossible jump: {ghost_df['ghost_jump'].sum():,}")

    check_dly_accuracy(df)

    route_summary = (
        matched.groupby("rt").agg(
            total_obs    =("vid",            "count"),
            avg_delay_min=("delay_minutes",  "mean"),
            pct_on_time  =("on_time_status", lambda x: (x == "on_time").mean() * 100),
            pct_early    =("on_time_status", lambda x: (x == "early").mean() * 100),
            pct_late     =("on_time_status", lambda x: (x == "late").mean() * 100),
        )
        .round(1)
        .sort_values("pct_on_time")
    )
    print(f"\n  Worst on-time performance by route:")
    print(route_summary.head(10).to_string())
    print(f"{'='*52}\n")

    # ── Write output ──────────────────────────────────────────────────────────
    out_cols = [
        "vid", "tmstmp", "rt", "des", "lat", "lon", "pdist", "dly",
        "matched_trip_id", "first_stop_name", "scheduled_departure_stop1",
        "schedule_elapsed_minutes", "prev_stop_name", "next_stop_name",
        "scheduled_arrival_next_minutes", "delay_minutes", "on_time_status",
        "dist_from_next_stop_km", "time_match_score_minutes",
        "pdist_match_used", "pdist_unreliable",
        "is_ghost", "ghost_score", "ghost_frozen",
        "ghost_never_departed", "ghost_jump",
    ]
    out_df   = df[[c for c in out_cols if c in df.columns]].copy()
    out_path = Path(output_path)

    if out_path.suffix == ".parquet" or out_path.is_dir():
        # Parquet partitioned by date + route — downstream queries skip
        # irrelevant partitions entirely (e.g. "give me route 66, all of March")
        out_path.mkdir(parents=True, exist_ok=True)
        out_df["date"] = pd.to_datetime(out_df["tmstmp"]).dt.date.astype(str)
        pq.write_to_dataset(
            pa.Table.from_pandas(out_df, preserve_index=False),
            root_path=str(out_path),
            partition_cols=["date", "rt"],
            compression="snappy",
            existing_data_behavior="overwrite_or_ignore",
        )
        _t(f"Parquet results written to {out_path}/", t0)
    else:
        out_df.to_csv(out_path, index=False)
        _t(f"CSV results written to {out_path}", t0)

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CTA Bus On-Time Analysis — optimized for large data volumes"
    )
    parser.add_argument("--vehicles",        default="Datasets/api_mar1.csv",
                        help="CSV/Parquet file or directory of files")
    parser.add_argument("--stop_times",      default="Datasets/stop_times.txt",
                        help="GTFS stop_times.txt (only needed with --preprocess-gtfs)")
    parser.add_argument("--trips",           default="Datasets/trips.txt")
    parser.add_argument("--stops",           default="Datasets/stops.txt")
    parser.add_argument("--calendar",        default="Datasets/calendar.txt",
                        help="GTFS calendar.txt for day-of-week filtering")
    parser.add_argument("--gtfs-cache",      default="./gtfs_cache",
                        help="Directory for pre-processed GTFS Parquet cache")
    parser.add_argument("--output",          default="metrics_results/bus_ontime_results_new.csv",
                        help="Output path — use a directory for partitioned Parquet output")
    parser.add_argument("--preprocess-gtfs", action="store_true",
                        help="Convert GTFS text files to Parquet cache (run once per feed update)")
    args = parser.parse_args()

    total_t0 = time.time()

    if args.preprocess_gtfs:
        preprocess_gtfs(args.stop_times, args.trips, args.stops, args.gtfs_cache,
                        calendar_path=args.calendar)
        if not Path(args.vehicles).exists():
            print("GTFS pre-processing done. "
                  "Run again without --preprocess-gtfs to analyse vehicles.")
            return

    vehicles     = load_vehicles(args.vehicles)
    vehicles     = detect_ghost_buses(vehicles)
    vehicles     = flag_unreliable_pdist(vehicles)

    route_filter  = set(vehicles["rt"].unique()) if FILTER_ROUTES else None
    stop_times, trips_cal = load_gtfs_cache(args.gtfs_cache, route_filter)
    candidate_index, route_index, pid_to_shapes = build_candidate_index(stop_times, trips_cal)

    vehicles     = match_vehicle_to_trip(vehicles, candidate_index, route_index, pid_to_shapes)
    vehicles     = compute_delay(vehicles)
    summarise_and_save(vehicles, args.output)

    print(f"Total runtime: {time.time() - total_t0:.1f}s")
    return vehicles


if __name__ == "__main__":
    main()