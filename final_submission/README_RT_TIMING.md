# route_timing.ipynb

Builds `route_stop_sequences.pkl` — a per-route dictionary of ordered stop sequences with empirically-estimated travel times between each consecutive stop pair. This file is a required input for the forward/backward imputation method in `data_imputation.ipynb`.

---

## Data Files

| File | Description |
|---|---|
| `data/all_data_final.csv` | Main bus GPS dataset (~8.2M rows). Columns include `vid` (vehicle ID), `rt` (route), `lat`/`lon`, `tmstmp`. |
| `Datasets/stops.txt` | GTFS stops — `stop_id`, `stop_name`, `stop_lat`, `stop_lon`. |
| `Datasets/trips.txt` | GTFS trips — `route_id`, `trip_id`. Used to map trips → routes and filter stop times to only relevant routes. |
| `Datasets/stop_times.txt` | GTFS stop times — `trip_id`, `stop_id`, `stop_sequence`, `arrival_time`. Used to (1) determine which stops are valid for each route, and (2) define the canonical stop order and scheduled travel times. |

---

## Output

```
Datasets/route_stop_sequences.pkl
```

A Python dict keyed by `route_id`. Each value is a DataFrame with one row per consecutive stop pair on that route:

| Column | Description |
|---|---|
| `stop_i` | Departing stop ID |
| `stop_j` | Arriving stop ID |
| `t_avg_to_next_min` | Estimated travel time from `stop_i` to `stop_j`, in minutes |

**To load:**
```python
import pickle
with open("Datasets/route_stop_sequences.pkl", "rb") as f:
    route_stop_sequences = pickle.load(f)
```

---

## How to Run

Run all cells top to bottom. The notebook is self-contained — no prior outputs are needed. The three main stages are Stop Matching, Empirical Travel Times, and Stop Sequence Construction.

---

## Stage 1 — Stop Matching

Each raw GPS observation is matched to its nearest scheduled stop using a `cKDTree` spatial index. A match is only made if the stop is actually served by the bus's route (via `stop_times.txt`) and falls within a radius of **0.0002 degrees (~22m)**.

The 22m radius was selected after testing three candidates (0.0001°, 0.0002°, 0.0003°) and checking what fraction of observations got matched at each radius. 0.0002° was chosen as the best balance between GPS accuracy and the risk of false matches at nearby stops. At this radius, approximately 37% of all observations are matched to a stop — meaning a bus is considered "at a stop" only when it's close enough to be credibly there, not just in the vicinity.

Unmatched observations (the remaining ~63%) are not discarded from the dataset — they just don't contribute to travel time estimates in the next stage.

---

## Stage 2 — Empirical Travel Times

Using only the stop-matched observations, consecutive stop arrivals from the same vehicle on the same route are paired up. For each pair `(stop_i → stop_j)`, the elapsed time in minutes is computed. Pairs are discarded if the travel time is outside the range **0.75–30 minutes** — times below this are likely duplicate pings or GPS noise, and times above this suggest the bus stopped running, went out of service, or the stop match was wrong.

The surviving pairs are averaged across all vehicles and all days to produce a single `t_avg_min` per `(route_id, stop_i, stop_j)` triple. This is the empirical travel time used in Stage 3.

---

## Stage 3 — Stop Sequence Construction

For each route, the **canonical stop sequence** is determined by finding the trip in GTFS with the most stops (i.e., the most complete trip). This trip's stop order defines the sequence that buses are assumed to follow.

For each consecutive stop pair in the sequence, a `t_avg_to_next_min` value is assigned using the following priority:

1. **Empirical** — if a directly observed average travel time exists for this `(stop_i, stop_j)` pair from Stage 2, use it.

2. **Interpolated** — if empirical data is missing for a run of consecutive stop pairs, but empirical anchors exist on both sides of the gap, the scheduled times from GTFS are used to distribute the total empirical gap proportionally across the missing pairs. This keeps absolute travel times grounded in real observations while using the schedule only for relative proportions.

3. **Scheduled** — if no empirical anchors exist nearby, the raw GTFS `arrival_time` difference between stops is used directly.

This hierarchy ensures that real observed speeds take precedence wherever data exists, with the GTFS schedule acting as a fallback rather than the primary source.

The final result is saved as `route_stop_sequences.pkl`.
