# CTA Bus Matching Algorithm

This notebook matches real-time CTA Bus Tracker API observations to their scheduled GTFS trips, computes per-observation delay, detects ghost buses, and produces a per-route performance table. It is the core analytics pipeline for the **CTA Bus Trust Score** project.

---

## Requirements

```
duckdb
numpy
pandas
pyarrow
```

Install with:
```bash
pip install duckdb numpy pandas pyarrow
```

---

## Input Data

Place the following files in a `Datasets/` folder before running:

| File | Source | Description |
|------|--------|-------------|
| `stop_times.txt` | CTA GTFS Static Feed | Scheduled arrival/departure times per stop per trip |
| `stops.txt` | CTA GTFS Static Feed | Stop locations and names |
| `trips.txt` | CTA GTFS Static Feed | Trip metadata (route, shape, direction, service ID) |
| `calendar.txt` | CTA GTFS Static Feed | Days of the week each service ID is active |
| `shapes.txt` | CTA GTFS Static Feed | Shape polylines for each trip (used in off-route detection) |
| `all_data_imputed_with_stops.csv` | CTA Bus Tracker API | Raw vehicle observations with imputed gaps |

GTFS files are available at the [CTA Developer Center](https://www.transitchicago.com/developers/gtfs/). API data is collected via the [CTA Bus Tracker API](https://www.transitchicago.com/developers/bustracker/).

---

## How to Run

The pipeline has two stages: **offline preprocessing** (run once) and **matching + analysis** (run each time).

### Step 1 — Preprocessing (run once)

```python
preprocess_static_data(stop_times_path, stops_path, trips_path, calendar_path)
preprocess_api_data(api_path)
```

These functions merge and cache the GTFS and API data as Snappy-compressed Parquet files in `gtfs_cache/` and `api_cache/`. This step is slow (~2 min for 8M rows) but only needs to run once per dataset. Comment them out after the first run.

### Step 2 — Load caches and build indexes

```python
stop_times, trips_calendar = load_gtfs_cache()
api = load_api_cache()

trip_store, shape_index = build_trip_index(stop_times, trips_calendar)
unique_pids = api['pid'].dropna().unique()
stst_index  = build_stst_index(trip_store, shape_index, unique_pids)
```

### Step 3 — Match, detect ghosts, and compute performance

```python
result = match_trips_bus(api, trip_store, shape_index, stst_index)
result = identify_ghost_buses(result, trip_store, trips_calendar, shapes_path='Datasets/shapes.txt')
perf   = analyze_route_performance(result)
```

### Step 4 — Save outputs

```python
perf.to_csv('metrics_results/performance_results.csv')
result.to_csv('metrics_results/mar1_analysis.csv')
```

---

## Pipeline Overview

```
GTFS Static Files ──► preprocess_static_data() ──► gtfs_cache/static_data.parquet
                                                     gtfs_cache/trips_calendar.parquet

API CSV ─────────────► preprocess_api_data() ────► api_cache/api_data.parquet

                            │
                            ▼
                   build_trip_index()       → trip_store, shape_index
                   build_stst_index()       → stst_index

                            │
                            ▼
                   match_trips_bus()        → result (with delay_minutes, match_type)

                            │
                            ▼
                   identify_ghost_buses()   → result (with ghost flags)

                            │
                            ▼
                   analyze_route_performance() → perf (per-route metrics CSV)
```

---

## Matching Algorithm

Each vehicle run is matched to a GTFS trip through three levels of fallback:

| Level | Method | Coverage |
|-------|--------|----------|
| 1 | **stst exact lookup** — matches on `(route_id, pid, scheduled_start_time)` with ±30s tolerance and a 50-minute sanity check | ~87.5% |
| 2 | **pid shape fallback** — scores all trips whose shape ends with the observed pid, filtered to the active weekday | ~9.9% |
| 3 | **full route fallback** — scores all trips on the route active on the observed weekday | ~0.3% |
| — | **unmatched** | ~2.0% |

Once a trip is matched, delay is computed per observation by linearly interpolating the scheduled time at the observed `pdist` position between the two bracketing stops.

---

## Ghost Bus Detection

Ghost buses are identified through four rules applied after matching:

| Rule | Criterion |
|------|-----------|
| **G1 — Freeze** | `pdist` unchanged for ≥ 20 consecutive minutes within a run |
| **G2 — Jump** | `pdist` changes by > 10,000 feet between consecutive pings within a run |
| **G3 — Off-route** | Vehicle position is > 2km from its matched shape polyline |
| **G4 — Never departed** | A scheduled trip has zero matched observations on a complete collection date |

G4 ghost trips are appended to the result DataFrame as synthetic rows with `match_type = 'ghost_trip'`. The `is_ghost` column is `True` for any observation flagged by G1–G4.

---

## Output Files

### `metrics_results/performance_results.csv`
One row per route, containing 60+ metrics including:

- `overall_otp_pct` — on-time performance (%)
- `overall_delay_mean/median/std/p90` — delay distribution (minutes)
- `overall_early_pct`, `overall_late_pct`, `overall_very_late_pct`
- `ghost_rate_pct` — never-departed trips as % of total scheduled trips
- All metrics repeated per time-of-day bucket: `early_morning`, `am_peak`, `midday`, `pm_peak`, `evening`

### `metrics_results/mar1_analysis.csv`
Full observation-level result with all original API columns plus:

- `matched_trip_id` — `{trip_id}_{trip_date}`
- `match_type` — `stst`, `fallback_pid`, `fallback_route`, `unscheduled`, or `ghost_trip`
- `delay_minutes` — positive = late, negative = early
- `schedule_elapsed_minutes` — time elapsed since scheduled first departure
- `prev_stop_name`, `next_stop_name` — the stops bracketing the observation
- `ghost_freeze`, `ghost_jump`, `ghost_offroute`, `is_ghost` — ghost flags
- `pre_trip` — `True` if the bus is idling before the trip has started

---

## Delay Thresholds

Delay classifications follow CTA standards:

| Category | Threshold |
|----------|-----------|
| Very Early | delay < −5 min |
| Early | −5 ≤ delay < −1 min |
| On Time | −1 ≤ delay ≤ 3 min |
| Slightly Late | 3 < delay ≤ 5 min |
| Late | 5 < delay ≤ 10 min |
| Very Late | 10 < delay ≤ 15 min |
| Severely Late | delay > 15 min |

---

## Project Context

This notebook is part of the **Assessing Chicago Transit Authority Bus System Reliability** project (DATA ENG 300, Northwestern University). See the full report for methodology details, imputation approach, and Bus Trust Score construction.
