"""
CTA Bus On-Time Analysis with Ghost Bus Detection
==================================================
Joins vehicle positions → stop_times → trips → stops using
position-based (pdist / shape_dist_traveled) fuzzy matching.

On-time window follows CTA's own standard:
    early   : delay < -1 min  (running ahead of schedule)
    on_time : -1 min ≤ delay ≤ +5 min
    late    : delay > +5 min

The API's `dly` field is used as an accuracy checker — after we classify
each vehicle, we compare our result against `dly` and report agreement rates
by route. This tells us how well our position-based matching is performing.

Ghost bus detection uses two physical signals (frozen pdist, impossible
position jump) plus a post-match coordinate sanity check. The API's `dly`
flag is intentionally excluded — a delayed bus is still a real bus.

Required files:
    - vehicles.csv      : Bus Tracker API snapshots (multiple time windows recommended)
    - stop_times.txt    : GTFS stop times
    - trips.txt         : GTFS trips
    - stops.txt         : GTFS stops

Expected stop_times.txt schema (standard GTFS):
    trip_id, arrival_time, departure_time, stop_id,
    stop_sequence, stop_headsign, pickup_type, shape_dist_traveled

Usage:
    python bus_ontime_analysis.py

    # Or with custom paths:
    python bus_ontime_analysis.py \
        --vehicles vehicles.csv \
        --stop_times stop_times.txt \
        --trips trips.txt \
        --stops stops.txt \
        --output results.csv
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

# Asymmetric on-time window matching CTA's own performance standard:
#   early   : delay < EARLY_THRESHOLD  (running ahead — bad for passengers)
#   on_time : EARLY_THRESHOLD ≤ delay ≤ LATE_THRESHOLD
#   late    : delay > LATE_THRESHOLD
EARLY_THRESHOLD_MINUTES = -1   # more than 1 min early = early
LATE_THRESHOLD_MINUTES  =  5   # more than 5 min late  = late

# Ghost bus detection thresholds
# NOTE: dly is intentionally excluded — a delayed bus is still a real bus.
GHOST_FROZEN_PDIST_THRESHOLD = 50    # feet; pdist change below this = possibly frozen
GHOST_MIN_OBSERVATIONS = 2           # need at least this many snapshots to flag frozen
GHOST_IMPOSSIBLE_JUMP_FEET = 5280    # 1 mile jump between polls = suspicious
GHOST_COORD_MISMATCH_KM = 5.0        # km; vehicle too far from matched stop segment = suspicious

# stop_times can be large — load only the routes we have vehicles for
FILTER_ROUTES = True  # set False to load all stop_times (slower)

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_gtfs_time(time_str):
    """
    Convert GTFS time string to total minutes since midnight.
    GTFS allows times past midnight (e.g. '25:30:00' for 1:30 AM next day).
    """
    try:
        h, m, s = map(int, str(time_str).split(":"))
        return h * 60 + m + s / 60
    except Exception:
        return np.nan


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def vehicle_time_to_minutes(tmstmp_series):
    """Convert vehicle timestamp to minutes since midnight (same day)."""
    dt = pd.to_datetime(tmstmp_series)
    return dt.dt.hour * 60 + dt.dt.minute + dt.dt.second / 60


# ── Step 1: Load & clean vehicles ─────────────────────────────────────────────

def load_vehicles(path):
    print(f"[1/6] Loading vehicles from {path}...")
    df = pd.read_csv(path)

    # Drop stale merge columns if vehicles.csv already has them
    drop_cols = ["trip_id", "schd_trip_id", "route_id", "tatripid_str"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df["tmstmp"] = pd.to_datetime(df["tmstmp"])
    df["dly"] = df["dly"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    df["rt"] = df["rt"].astype(str).str.strip()
    df["pdist"] = pd.to_numeric(df["pdist"], errors="coerce")
    df["vid"] = df["vid"].astype(str)
    df["time_minutes"] = vehicle_time_to_minutes(df["tmstmp"])

    print(f"    {len(df)} vehicle observations | {df['vid'].nunique()} unique vehicles | {df['rt'].nunique()} routes")
    return df


# ── Step 2: Ghost bus detection ───────────────────────────────────────────────

def detect_ghost_buses(vehicles):
    """
    Flag potential ghost buses using two physical signals:
      G1 - FROZEN: same vid reports nearly identical pdist across multiple snapshots
      G2 - JUMP:   pdist moves impossibly fast between consecutive observations

    Note: the API's `dly` field is NOT used here — a delayed bus is still a
    real bus. dly is instead used later as an accuracy checker for our delay
    classifications.

    Returns vehicles df with new columns:
      ghost_frozen, ghost_jump, ghost_score (0–2), is_ghost
    """
    print("[2/6] Detecting ghost buses...")
    df = vehicles.copy().sort_values(["vid", "tmstmp"])

    # G1 & G2: require multiple observations per vid
    if df.groupby("vid").size().max() >= GHOST_MIN_OBSERVATIONS:
        df["_prev_pdist"] = df.groupby("vid")["pdist"].shift(1)
        df["_prev_tmstmp"] = df.groupby("vid")["tmstmp"].shift(1)
        df["_pdist_delta"] = (df["pdist"] - df["_prev_pdist"]).abs()
        df["_time_delta_min"] = (df["tmstmp"] - df["_prev_tmstmp"]).dt.total_seconds() / 60

        # G1: frozen — pdist barely moved relative to time elapsed
        df["ghost_frozen"] = (
            (df["_pdist_delta"] < GHOST_FROZEN_PDIST_THRESHOLD) &
            (df["_time_delta_min"] > 1) &   # at least 1 min has passed
            df["_prev_pdist"].notna()
        )

        # G2: impossible jump — moved more than threshold feet between polls
        df["ghost_jump"] = (
            (df["_pdist_delta"] > GHOST_IMPOSSIBLE_JUMP_FEET) &
            df["_prev_pdist"].notna()
        )
    else:
        print("    ⚠ Only one snapshot per vehicle — frozen/jump detection skipped (need multiple polls)")
        df["ghost_frozen"] = False
        df["ghost_jump"] = False

    df["ghost_score"] = (
        df["ghost_frozen"].astype(int) +
        df["ghost_jump"].astype(int)
    )
    # A single signal is enough here since dly is no longer a tiebreaker
    df["is_ghost"] = df["ghost_score"] >= 1

    # Clean up temp cols
    df = df.drop(columns=["_prev_pdist", "_prev_tmstmp", "_pdist_delta", "_time_delta_min"], errors="ignore")

    n_ghost = df[df["is_ghost"]]["vid"].nunique()
    n_frozen = df[df["ghost_frozen"]]["vid"].nunique()
    n_jump = df[df["ghost_jump"]]["vid"].nunique()
    print(f"    Ghost buses flagged: {n_ghost} vehicles")
    print(f"      → Frozen pdist:     {n_frozen} vehicles")
    print(f"      → Impossible jumps: {n_jump} vehicles")

    return df


# ── Step 3: Load GTFS ─────────────────────────────────────────────────────────

def load_gtfs(stop_times_path, trips_path, stops_path, route_filter=None):
    print(f"[3/6] Loading GTFS files...")

    trips = pd.read_csv(trips_path, dtype={"route_id": str, "trip_id": str, "schd_trip_id": str})
    stops = pd.read_csv(stops_path, dtype={"stop_id": str})
    print(f"    trips: {len(trips):,} rows | stops: {len(stops):,} rows")

    # Filter trips to only routes we care about (speeds up stop_times load)
    if route_filter:
        trips_filtered = trips[trips["route_id"].isin(route_filter)]
        trip_ids_needed = set(trips_filtered["trip_id"].astype(str))
        print(f"    Filtering stop_times to {len(trip_ids_needed):,} trip_ids across {len(route_filter)} routes...")
    else:
        trips_filtered = trips
        trip_ids_needed = None

    # Load stop_times in chunks to handle large files
    print(f"    Loading stop_times.txt (this may take a moment)...")
    chunks = []
    for chunk in pd.read_csv(
        stop_times_path,
        dtype={"trip_id": str, "stop_id": str},
        chunksize=500_000
    ):
        if trip_ids_needed:
            chunk = chunk[chunk["trip_id"].isin(trip_ids_needed)]
        chunks.append(chunk)
    stop_times = pd.concat(chunks, ignore_index=True)
    print(f"    stop_times loaded: {len(stop_times):,} rows")

    # Parse scheduled times to minutes since midnight
    stop_times["arrival_minutes"] = stop_times["arrival_time"].apply(parse_gtfs_time)
    stop_times["departure_minutes"] = stop_times["departure_time"].apply(parse_gtfs_time)
    stop_times["shape_dist_traveled"] = pd.to_numeric(stop_times["shape_dist_traveled"], errors="coerce")
    stop_times = stop_times.dropna(subset=["shape_dist_traveled", "arrival_minutes"])
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

    # Join stop names/coords onto stop_times
    stop_times = stop_times.merge(
        stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
        on="stop_id", how="left"
    )

    # Join route_id onto stop_times via trips
    stop_times = stop_times.merge(
        trips_filtered[["trip_id", "route_id", "direction", "schd_trip_id"]],
        on="trip_id", how="left"
    )
    stop_times["route_id"] = stop_times["route_id"].astype(str).str.strip()

    return stop_times, trips_filtered, stops


# ── Step 4: Position-based trip matching ──────────────────────────────────────

def match_vehicle_to_trip(vehicles, stop_times):
    """
    For each vehicle observation, find the best matching scheduled trip by:
      1. Same route (vehicles.rt == stop_times.route_id)
      2. The scheduled stop bracket whose shape_dist_traveled range
         contains the vehicle's pdist (i.e. vehicle is between stop N and N+1)
      3. Scheduled time at those stops is close to the vehicle's current time

    Returns vehicles with new columns:
      matched_trip_id, prev_stop_id, prev_stop_name, prev_stop_dist,
      next_stop_id, next_stop_name, next_stop_dist,
      scheduled_arrival_next, delay_minutes, on_time_status
    """
    print("[4/6] Matching vehicles to scheduled trips by position + time...")

    # Pre-group stop_times by route for fast lookup
    st_by_route = {rt: grp for rt, grp in stop_times.groupby("route_id")}

    results = []
    unmatched = 0

    for _, veh in vehicles.iterrows():
        rt = str(veh["rt"]).strip()
        pdist = veh["pdist"]
        veh_time = veh["time_minutes"]

        if rt not in st_by_route or pd.isna(pdist):
            unmatched += 1
            results.append({})
            continue

        route_st = st_by_route[rt]

        # For each trip on this route, find the stop bracket containing pdist
        # and score by how close the scheduled time is to the vehicle's current time
        best_match = None
        best_score = np.inf

        for trip_id, trip_stops in route_st.groupby("trip_id"):
            trip_stops = trip_stops.sort_values("stop_sequence")
            dists = trip_stops["shape_dist_traveled"].values

            # Find where pdist falls in this trip's stop sequence
            idx = np.searchsorted(dists, pdist, side="right") - 1

            # Must be a valid interior bracket (not before first or after last stop)
            if idx < 0 or idx >= len(trip_stops) - 1:
                continue

            prev_stop = trip_stops.iloc[idx]
            next_stop = trip_stops.iloc[idx + 1]

            # Interpolate expected scheduled time at vehicle's current pdist
            dist_range = next_stop["shape_dist_traveled"] - prev_stop["shape_dist_traveled"]
            if dist_range <= 0:
                continue
            frac = (pdist - prev_stop["shape_dist_traveled"]) / dist_range
            expected_sched_time = (
                prev_stop["departure_minutes"] +
                frac * (next_stop["arrival_minutes"] - prev_stop["departure_minutes"])
            )

            # Score = absolute difference between vehicle's clock and scheduled clock
            score = abs(veh_time - expected_sched_time)

            if score < best_score:
                best_score = score
                best_match = {
                    "matched_trip_id": trip_id,
                    "prev_stop_id": prev_stop["stop_id"],
                    "prev_stop_name": prev_stop.get("stop_name", ""),
                    "prev_stop_dist": prev_stop["shape_dist_traveled"],
                    "next_stop_id": next_stop["stop_id"],
                    "next_stop_name": next_stop.get("stop_name", ""),
                    "next_stop_dist": next_stop["shape_dist_traveled"],
                    "next_stop_lat": next_stop.get("stop_lat", np.nan),
                    "next_stop_lon": next_stop.get("stop_lon", np.nan),
                    "scheduled_arrival_next_minutes": next_stop["arrival_minutes"],
                    "scheduled_arrival_next": next_stop["arrival_time"],
                    "time_match_score_minutes": round(best_score, 2),
                }

        if best_match is None:
            unmatched += 1
            results.append({})
        else:
            results.append(best_match)

    print(f"    Matched: {len(vehicles) - unmatched} / {len(vehicles)} vehicles")

    match_df = pd.DataFrame(results, index=vehicles.index)
    return pd.concat([vehicles, match_df], axis=1)


# ── Step 5: Compute delay ─────────────────────────────────────────────────────

def compute_delay(df):
    """
    delay_minutes = vehicle's current time - interpolated scheduled time at pdist position
    Positive = late, Negative = early.

    On-time classification follows CTA's standard:
      early   : delay < EARLY_THRESHOLD_MINUTES  (-1 min)
      on_time : EARLY_THRESHOLD_MINUTES ≤ delay ≤ LATE_THRESHOLD_MINUTES  (-1 to +5 min)
      late    : delay > LATE_THRESHOLD_MINUTES   (+5 min)
    """
    print("[5/6] Computing delay...")

    df["delay_minutes"] = df["time_minutes"] - df["scheduled_arrival_next_minutes"]

    # Handle overnight wrap (e.g. vehicle at 23:58, scheduled at 00:02)
    df.loc[df["delay_minutes"] < -720, "delay_minutes"] += 1440
    df.loc[df["delay_minutes"] > 720, "delay_minutes"] -= 1440

    df["delay_minutes"] = df["delay_minutes"].round(1)

    def classify(row):
        if pd.isna(row.get("matched_trip_id")):
            return "unmatched"
        if row.get("is_ghost"):
            return "ghost"
        d = row["delay_minutes"]
        if pd.isna(d):
            return "unmatched"
        if d < EARLY_THRESHOLD_MINUTES:
            return "early"
        if d <= LATE_THRESHOLD_MINUTES:
            return "on_time"
        return "late"

    df["on_time_status"] = df.apply(classify, axis=1)

    # G3 ghost signal (post-match): vehicle coords too far from matched stop segment
    has_coords = df["next_stop_lat"].notna() & df["lat"].notna()
    df.loc[has_coords, "dist_from_next_stop_km"] = haversine_km(
        df.loc[has_coords, "lat"],
        df.loc[has_coords, "lon"],
        df.loc[has_coords, "next_stop_lat"],
        df.loc[has_coords, "next_stop_lon"],
    ).round(3)

    coord_ghost = (
        df["dist_from_next_stop_km"] > GHOST_COORD_MISMATCH_KM
    ) & ~df["is_ghost"]
    df.loc[coord_ghost, "is_ghost"] = True
    df.loc[coord_ghost, "ghost_score"] += 1
    df.loc[coord_ghost, "on_time_status"] = "ghost"

    return df


# ── Step 6: Summarise & output ────────────────────────────────────────────────

def check_dly_accuracy(df):
    """
    Use the API's `dly` field as an accuracy checker for our delay classifications.

    Logic:
      - dly=True  should correspond to our "late" classification
      - dly=False should correspond to our "on_time" or "early" classification

    We report agreement rates overall and per route so we can spot where our
    position-based matching is performing poorly.
    """
    # Only check matched, non-ghost vehicles where dly is meaningful
    check = df[
        df["on_time_status"].isin(["on_time", "early", "late"]) &
        df["dly"].notna()
    ].copy()

    if len(check) == 0:
        print("    ⚠ No matched vehicles to check dly accuracy against.")
        return None

    # Agreement: dly=True AND we said late → agree; dly=False AND we said on_time/early → agree
    check["dly_says_late"] = check["dly"].astype(bool)
    check["we_say_late"] = check["on_time_status"] == "late"
    check["agree"] = check["dly_says_late"] == check["we_say_late"]

    overall_agreement = check["agree"].mean() * 100

    # Disagreement breakdown
    false_positives = check[check["dly_says_late"] & ~check["we_say_late"]]   # dly=True but we say on time
    false_negatives = check[~check["dly_says_late"] & check["we_say_late"]]   # dly=False but we say late

    print(f"\n  dly ACCURACY CHECK (vs our delay classifications)")
    print(f"  {'─'*46}")
    print(f"  Overall agreement:      {overall_agreement:.1f}%  ({check['agree'].sum()} / {len(check)} vehicles)")
    print(f"  API says late, we don't: {len(false_positives)} vehicles  "
          f"({len(false_positives)/len(check)*100:.1f}%)  ← possible under-detection")
    print(f"  We say late, API doesn't: {len(false_negatives)} vehicles  "
          f"({len(false_negatives)/len(check)*100:.1f}%)  ← possible over-detection")

    # Per-route agreement
    route_agreement = (
        check.groupby("rt")
        .agg(
            n=("agree", "count"),
            agreement_pct=("agree", lambda x: x.mean() * 100),
            fp_count=("agree", lambda x: ((check.loc[x.index, "dly_says_late"]) & ~(check.loc[x.index, "we_say_late"])).sum()),
            fn_count=("agree", lambda x: (~(check.loc[x.index, "dly_says_late"]) & (check.loc[x.index, "we_say_late"])).sum()),
        )
        .round(1)
        .sort_values("agreement_pct")
    )

    print(f"\n  Routes with lowest dly agreement (potential matching issues):")
    print(route_agreement.head(10).to_string())
    route_agreement.to_csv('metrics_results/delay_accuracy.csv')

    return route_agreement


def summarise(df, output_path):
    print("[6/6] Summarising results...")

    status_counts = df["on_time_status"].value_counts()
    total = len(df)

    print(f"\n{'='*50}")
    print(f"  ON-TIME SUMMARY  ({total} vehicle observations)")
    print(f"  Window: <{EARLY_THRESHOLD_MINUTES} min = early | "
          f"{EARLY_THRESHOLD_MINUTES} to +{LATE_THRESHOLD_MINUTES} min = on time | "
          f">{LATE_THRESHOLD_MINUTES} min = late")
    print(f"{'='*50}")
    for status, count in status_counts.items():
        pct = count / total * 100
        print(f"  {status:<12}: {count:>4}  ({pct:.1f}%)")

    matched = df[df["on_time_status"].isin(["on_time", "early", "late"])]
    if len(matched):
        print(f"\n  Avg delay (matched, non-ghost): {matched['delay_minutes'].mean():.1f} min")
        print(f"  Median delay:                   {matched['delay_minutes'].median():.1f} min")

    ghost_df = df[df["is_ghost"]]
    if len(ghost_df):
        print(f"\n  Ghost buses: {ghost_df['vid'].nunique()} unique vehicles")
        print(f"  Ghost signals breakdown:")
        print(f"    Frozen pdist:     {ghost_df['ghost_frozen'].sum()}")
        print(f"    Impossible jump:  {ghost_df['ghost_jump'].sum()}")
        coord_ghost_count = (ghost_df["dist_from_next_stop_km"] > GHOST_COORD_MISMATCH_KM).sum()
        print(f"    Coord mismatch:   {coord_ghost_count}")

    # dly accuracy check
    check_dly_accuracy(df)

    print(f"\n{'='*50}")

    # Per-route on-time summary
    route_summary = (
        matched
        .groupby("rt")
        .agg(
            total_obs=("vid", "count"),
            avg_delay_min=("delay_minutes", "mean"),
            pct_on_time=("on_time_status", lambda x: (x == "on_time").mean() * 100),
            pct_early=("on_time_status", lambda x: (x == "early").mean() * 100),
            pct_late=("on_time_status", lambda x: (x == "late").mean() * 100),
        )
        .round(1)
        .sort_values("pct_on_time")
    )
    print("\n  Worst on-time performance by route:")
    print(route_summary.head(10).to_string())
    print()

    # Output columns
    out_cols = [
        "vid", "tmstmp", "rt", "des", "lat", "lon", "pdist", "dly",
        "matched_trip_id", "prev_stop_name", "next_stop_name",
        "scheduled_arrival_next", "delay_minutes", "on_time_status",
        "dist_from_next_stop_km", "time_match_score_minutes",
        "is_ghost", "ghost_score", "ghost_frozen", "ghost_jump",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(output_path, index=False)
    print(f"  Results saved to: {output_path}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CTA Bus On-Time Analysis")
    parser.add_argument("--vehicles",   default="vehicles.csv")
    parser.add_argument("--stop_times", default="stop_times.txt")
    parser.add_argument("--trips",      default="trips.txt")
    parser.add_argument("--stops",      default="stops.txt")
    parser.add_argument("--output",     default="metrics_results/bus_ontime_results.csv")
    args = parser.parse_args()

    # Load vehicles
    vehicles = load_vehicles(args.vehicles)

    # Ghost detection
    vehicles = detect_ghost_buses(vehicles)

    # Routes to filter GTFS by
    route_filter = set(vehicles["rt"].unique()) if FILTER_ROUTES else None

    # Load GTFS
    stop_times, trips, stops = load_gtfs(
        args.stop_times, args.trips, args.stops, route_filter
    )

    # Match positions to trips
    vehicles = match_vehicle_to_trip(vehicles, stop_times)

    # Compute delay
    vehicles = compute_delay(vehicles)

    # Summarise & save
    summarise(vehicles, args.output)

    return vehicles


if __name__ == "__main__":
    main()


'''
python claude_solution.py \
  --vehicles data/bus_data_current_chicago.csv \
  --stop_times Datasets/stop_times.txt \
  --trips Datasets/trips.txt \
  --stops Datasets/stops.txt  
'''