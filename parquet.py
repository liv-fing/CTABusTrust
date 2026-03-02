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
# Speed-based jump detection: flag if implied speed exceeds this threshold.
# 60 mph = 5,280 ft/min. Express buses top out around 55 mph on a good day.
# We use 90 mph (7,920 ft/min) to give a comfortable margin above any real
# bus speed while still catching GPS teleportation events.
GHOST_MAX_SPEED_FT_PER_MIN  = 7920   # 90 mph in feet per minute

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

def preprocess_gtfs(stop_times_path, trips_path, stops_path, cache_dir):
    """
    Merge stop_times + trips + stops into one enriched Parquet file.
    Run once when you get a new GTFS feed. Loads from cache on all future runs.

    Output: {cache_dir}/stop_times_enriched.parquet
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "stop_times_enriched.parquet"

    print(f"\n[GTFS PRE-PROCESS] Building {out_path} ...")
    t0 = time.time()

    trips = pd.read_csv(trips_path,
                        dtype={"route_id": str, "trip_id": str, "schd_trip_id": str})
    stops = pd.read_csv(stops_path, dtype={"stop_id": str})
    print(f"    trips: {len(trips):,} rows | stops: {len(stops):,} rows")

    # Load stop_times in chunks — it's large
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

    # Parse times, enrich with stops and route info
    st["arrival_minutes"]     = st["arrival_time"].apply(parse_gtfs_time)
    st["departure_minutes"]   = st["departure_time"].apply(parse_gtfs_time)
    st["shape_dist_traveled"] = pd.to_numeric(st["shape_dist_traveled"], errors="coerce")
    st = st.dropna(subset=["shape_dist_traveled", "arrival_minutes"])
    st = st.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)

    st = st.merge(stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
                  on="stop_id", how="left")
    st = st.merge(trips[["trip_id", "route_id", "direction", "schd_trip_id"]],
                  on="trip_id", how="left")
    st["route_id"] = st["route_id"].astype(str).str.strip()

    keep = ["trip_id", "stop_id", "stop_sequence", "arrival_minutes",
            "departure_minutes", "shape_dist_traveled", "stop_name",
            "stop_lat", "stop_lon", "route_id", "direction", "schd_trip_id"]
    st = st[[c for c in keep if c in st.columns]]
    st = st.sort_values(["route_id", "trip_id", "stop_sequence"])

    pq.write_table(
        pa.Table.from_pandas(st, preserve_index=False),
        out_path,
        row_group_size=100_000,
        compression="snappy",
    )
    size_mb = out_path.stat().st_size / 1_048_576
    _t(f"Parquet written ({size_mb:.1f} MB) -> {out_path}", t0)
    print(f"[GTFS PRE-PROCESS] Complete. Total: {time.time() - t0:.1f}s\n")
    return str(out_path)


def load_gtfs_cache(cache_dir, route_filter=None):
    """
    Load enriched stop_times from Parquet.
    DuckDB pushes the route filter down into the file scan so only matching
    row groups are read — much faster than loading everything then filtering.
    """
    out_path = Path(cache_dir) / "stop_times_enriched.parquet"
    if not out_path.exists():
        raise FileNotFoundError(
            f"GTFS cache not found at {out_path}. "
            "Run with --preprocess-gtfs first."
        )

    t0 = time.time()
    con = duckdb.connect()
    if route_filter:
        routes_sql = ", ".join(f"'{r}'" for r in route_filter)
        st = con.execute(
            f"SELECT * FROM read_parquet('{out_path}') "
            f"WHERE route_id IN ({routes_sql})"
        ).df()
    else:
        st = con.execute(f"SELECT * FROM read_parquet('{out_path}')").df()
    con.close()

    _t(f"GTFS cache loaded: {len(st):,} stop_times", t0)
    return st


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

    G1 - SUSTAINED FREEZE: same vid reports identical pdist for 20+ consecutive
         minutes. Short stops at lights or terminals are normal; only a prolonged
         freeze indicates the bus has stopped transmitting real position data.

    G2 - IMPOSSIBLE FORWARD JUMP: pdist increases by more than 1 mile in a single
         poll interval. Backward jumps (pdist resets to 0) are intentionally
         ignored — they indicate a new trip run, not a ghost.

    Both signals are evaluated per (vid, rt) pair so a vehicle switching routes
    doesn't produce false jumps.

    Note: dly is not a ghost signal — a delayed bus is still a real bus.
    """
    print("[2/6] Detecting ghost buses...")
    t0 = time.time()
    df = vehicles.copy().sort_values(["vid", "rt", "tmstmp"])

    if df.groupby("vid").size().max() >= GHOST_MIN_OBSERVATIONS:
        g = df.groupby(["vid", "rt"])

        df["_prev_pdist"]     = g["pdist"].shift(1)
        df["_prev_tmstmp"]    = g["tmstmp"].shift(1)
        df["_pdist_delta"]    = df["pdist"] - df["_prev_pdist"]  # signed
        df["_time_delta_min"] = (
            (df["tmstmp"] - df["_prev_tmstmp"]).dt.total_seconds() / 60
        )

        # Flag rows immediately following a trip reset so we can exclude them
        # from jump detection. A reset is when pdist drops significantly.
        df["_prev_was_reset"] = df["_pdist_delta"] < -GHOST_MAX_SPEED_FT_PER_MIN
        df["ghost_frozen"] = (
            (df["_pdist_delta"].abs() < GHOST_FROZEN_PDIST_THRESHOLD)
            & (df["_time_delta_min"] >= GHOST_FROZEN_MIN_MINUTES)
            & df["_prev_pdist"].notna()
        )

        # G2: impossible forward speed — catches GPS teleportation while
        # allowing express buses to move quickly between stops.
        # Normalize by time so a 2-minute poll interval gets 2x the allowance.
        df["_implied_speed"] = df["_pdist_delta"] / df["_time_delta_min"].clip(lower=0.5)
        df["ghost_jump"] = (
            (df["_implied_speed"] > GHOST_MAX_SPEED_FT_PER_MIN)
            & ~df["_prev_was_reset"].shift(-1).fillna(False)
            & df["_prev_pdist"].notna()
            & (df["_time_delta_min"] > 0)
        )

        df = df.drop(columns=["_prev_pdist", "_prev_tmstmp", "_pdist_delta",
                               "_time_delta_min", "_prev_was_reset", "_implied_speed"])
    else:
        print("    ⚠ Single snapshot per vehicle — frozen/jump detection skipped")
        df["ghost_frozen"] = False
        df["ghost_jump"]   = False

    df["ghost_score"] = df["ghost_frozen"].astype(int) + df["ghost_jump"].astype(int)

    # Frozen alone is sufficient for a sustained freeze.
    # Jump alone is not — a single bad GPS ping doesn't confirm a ghost.
    df["is_ghost"] = df["ghost_frozen"] | (df["ghost_score"] >= 2)

    _t(f"Ghost buses: {df[df['is_ghost']]['vid'].nunique()} vehicles "
       f"(frozen={df['ghost_frozen'].sum():,}, jump={df['ghost_jump'].sum():,})", t0)
    return df


# ── Step 3: Build vectorized trip index ───────────────────────────────────────

def build_trip_index(stop_times):
    """
    Convert the stop_times DataFrame into a dict of numpy arrays, keyed by
    route_id. Done once up front so the matcher never touches a DataFrame
    inside its inner loop.

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
    index   = {}
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


def match_vehicle_to_trip(vehicles, trip_index):
    """
    Vectorized position + time matching for vehicles with reliable pdist.
    Time-only fallback matching for vehicles with unreliable pdist.
    Processes vehicles in batches per route for memory efficiency.
    """
    print("[4/6] Matching vehicles to scheduled trips (vectorized)...")
    t0 = time.time()

    result_rows = [None] * len(vehicles)
    unmatched   = 0

    for rt, rt_vehicles in vehicles.groupby("rt"):
        rt = str(rt).strip()
        trips_for_route = trip_index.get(rt)

        if not trips_for_route:
            for idx in rt_vehicles.index:
                result_rows[idx] = {}
            unmatched += len(rt_vehicles)
            continue

        # Split into reliable and unreliable pdist groups
        reliable   = rt_vehicles[~rt_vehicles["pdist_unreliable"]]
        unreliable = rt_vehicles[rt_vehicles["pdist_unreliable"]]

        # ── Vectorized position+time match for reliable pdist ─────────────
        if len(reliable) > 0:
            indices = reliable.index.tolist()
            pdists  = reliable["pdist"].values.astype(np.float32)
            times   = reliable["time_minutes"].values.astype(np.float32)

            for start in range(0, len(indices), MATCH_BATCH_SIZE):
                end          = min(start + MATCH_BATCH_SIZE, len(indices))
                batch_idx    = indices[start:end]
                batch_pdists = pdists[start:end]
                batch_times  = times[start:end]

                valid = ~np.isnan(batch_pdists)
                batch_pdists_clean = np.where(valid, batch_pdists, np.float32(-1.0))

                best_trip_idx, best_bracket, best_scores = _match_batch(
                    batch_pdists_clean, batch_times, trips_for_route
                )

                for i, orig_idx in enumerate(batch_idx):
                    if not valid[i] or best_trip_idx[i] == -1:
                        result_rows[orig_idx] = {}
                        unmatched += 1
                        continue

                    trip = trips_for_route[best_trip_idx[i]]
                    b    = best_bracket[i]
                    frac = ((batch_pdists[i] - trip["dists"][b]) /
                            (trip["dists"][b + 1] - trip["dists"][b]))

                    result_rows[orig_idx] = {
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
                        "pdist_match_used":               True,
                        "delay_minutes":                  round(float(batch_times[i] - (
                            trip["dep_minutes"][b] +
                            frac * (trip["arr_minutes"][b + 1] - trip["dep_minutes"][b])
                        )), 1),
                        "time_match_score_minutes":       round(float(best_scores[i]), 2),
                    }

        # ── Time-only fallback for unreliable pdist ───────────────────────
        for orig_idx, row in unreliable.iterrows():
            trip, b, score = _time_only_match(
                row["time_minutes"], trips_for_route
            )
            if trip is None:
                result_rows[orig_idx] = {}
                unmatched += 1
                continue

            result_rows[orig_idx] = {
                "matched_trip_id":                trip["trip_id"],
                "prev_stop_id":                   trip["stop_ids"][b],
                "prev_stop_name":                 trip["stop_names"][b],
                "next_stop_id":                   trip["stop_ids"][b + 1],
                "next_stop_name":                 trip["stop_names"][b + 1],
                "next_stop_lat":                  float(trip["stop_lats"][b + 1]),
                "next_stop_lon":                  float(trip["stop_lons"][b + 1]),
                "scheduled_arrival_next_minutes": float(trip["arr_minutes"][b + 1]),
                "pdist_match_used":               False,
                # Delay uses midpoint of the scheduled bracket — less precise
                # than position-based but avoids the corrupted pdist entirely
                "delay_minutes":                  round(float(row["time_minutes"] - (
                    (trip["dep_minutes"][b] + trip["arr_minutes"][b + 1]) / 2
                )), 1),
                "time_match_score_minutes":       round(float(score), 2),
            }

    matched = len(vehicles) - unmatched
    _t(f"Matched {matched:,} / {len(vehicles):,} vehicles", t0)

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
    d        = df["delay_minutes"]
    score    = df["time_match_score_minutes"]

    # Unscheduled: high match score but delay is within the on-time window.
    # If the delay itself is large the bus is just late — don't filter it out.
    unscheduled = (
        matched & ~is_ghost
        & (score > MAX_MATCH_SCORE_MINUTES)
        & (d.abs() <= MAX_MATCH_SCORE_MINUTES)
    )

    df["on_time_status"] = np.select(
        condlist=[
            ~matched,
            is_ghost,
            unscheduled,
            matched & ~is_ghost & ~unscheduled & (d < EARLY_THRESHOLD_MINUTES),
            matched & ~is_ghost & ~unscheduled & (d >= EARLY_THRESHOLD_MINUTES) & (d <= LATE_THRESHOLD_MINUTES),
            matched & ~is_ghost & ~unscheduled & (d > LATE_THRESHOLD_MINUTES),
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

    Reports overall agreement and per-route breakdown sorted worst-first.
    Low agreement on a route is a signal that the position matching needs
    investigation for that route.
    """
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
        "matched_trip_id", "prev_stop_name", "next_stop_name",
        "scheduled_arrival_next_minutes", "delay_minutes", "on_time_status",
        "dist_from_next_stop_km", "time_match_score_minutes",
        "pdist_match_used", "pdist_unreliable",
        "is_ghost", "ghost_score", "ghost_frozen", "ghost_jump",
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
    parser.add_argument("--gtfs-cache",      default="./gtfs_cache",
                        help="Directory for pre-processed GTFS Parquet cache")
    parser.add_argument("--output",          default="bus_ontime_results.csv",
                        help="Output path — use a directory for partitioned Parquet output")
    parser.add_argument("--preprocess-gtfs", action="store_true",
                        help="Convert GTFS text files to Parquet cache (run once per feed update)")
    args = parser.parse_args()

    total_t0 = time.time()

    if args.preprocess_gtfs:
        preprocess_gtfs(args.stop_times, args.trips, args.stops, args.gtfs_cache)
        if not Path(args.vehicles).exists():
            print("GTFS pre-processing done. "
                  "Run again without --preprocess-gtfs to analyse vehicles.")
            return

    vehicles     = load_vehicles(args.vehicles)
    vehicles     = detect_ghost_buses(vehicles)
    vehicles     = flag_unreliable_pdist(vehicles)

    route_filter = set(vehicles["rt"].unique()) if FILTER_ROUTES else None
    stop_times   = load_gtfs_cache(args.gtfs_cache, route_filter)
    trip_index   = build_trip_index(stop_times)

    vehicles     = match_vehicle_to_trip(vehicles, trip_index)
    vehicles     = compute_delay(vehicles)
    summarise_and_save(vehicles, args.output)

    print(f"Total runtime: {time.time() - total_t0:.1f}s")
    return vehicles


if __name__ == "__main__":
    main()