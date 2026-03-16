# data_imputation.ipynb

Compares three strategies for filling a real gap in CTA bus GPS data, then applies the best one. Exports a single combined CSV ready for downstream analysis.

---

## Data Files

| File | Description |
|---|---|
| `data/all_data_final.csv` | Main bus GPS dataset (~8.2M rows). Columns include `vid` (vehicle ID), `rt` (route), `lat`/`lon`, `tmstmp` (timestamp). |
| `Datasets/route_stop_sequences.pkl` | Precomputed dict mapping each route ID to its ordered list of stop IDs. Used by the forward/backward fill method to walk a bus along its route. |
| `Datasets/stops.txt` | GTFS stops — `stop_id`, `stop_name`, `stop_lat`, `stop_lon`. |
| `Datasets/trips.txt` | GTFS trips — `route_id`, `trip_id`. Used to map trips → routes. |
| `Datasets/stop_times.txt` | GTFS stop times — `trip_id`, `stop_id`, `departure_time`. Used to find which stops belong to a route, and by the GTFS imputation method for scheduled departure times. |

---

## How to Run

1. Set the working directory to the project root (where `data/` and `Datasets/` live).
2. Run all cells top to bottom. The notebook is divided into:
   - **Setup** — imports and data loading
   - **Functions** — all helpers and imputation methods defined here
   - **Testing** — methods evaluated on a synthetic 2-hour gap
   - **Imputing Real Gap** — applies the chosen method to the actual gap and exports output

---

## The Real Gap

The dataset has a single large gap detected by `detect_gaps`:

- **Start:** 2026-02-24 04:05 CST  
- **End:** 2026-02-24 09:38 CST  
- **Duration:** ~333 minutes (~5.5 hours)

---

## Imputation Methods

### 1. `impute_gap_fill` — Same-Week Fill

**The idea:** The same buses running the same routes at 4am on a Tuesday will look almost identical the following Tuesday. Copy those real observations and shift their timestamps back by exactly 7 days.

**How it works:**
1. Call `identify_missing_buses` to find all routes/vehicles that were active within 2 hours on either side of the gap. These are the "active routes" — we only impute buses we know were actually running.
2. Pull all observations from exactly one week later that fall within the same time window and belong to one of those active routes.
3. Subtract 7 days from every timestamp in that slice.
4. Tag the resulting rows with `source = "imputed_same_period"`.

**Result on real gap:** 179,130 rows imputed, covering 116/116 active routes.

**Limitation:** Requires a clean week of data after the gap. If the following week also has missing data, this method fails silently (returns fewer rows or misses routes).

---

### 2. `impute_gap_fwd_bwd` — Route-Walking Forward/Backward Fill

**The idea:** We know where each bus was just before (or just after) the gap, and we know the ordered list of stops on its route. We can simulate where the bus *would have been* during the gap by walking it forward (or backward) through its stop sequence at its observed average speed.

**How it works:**

First, `identify_missing_buses` classifies each vehicle into one of three groups:
- `both_sides` — seen within 2h before AND within 2h after the gap
- `before_only` — only seen before the gap
- `after_only` — only seen after the gap

**Forward fill** (used for `both_sides` and `before_only` buses):
1. Take the bus's last known stop before the gap and the timestamp it was there.
2. Look up the stop's position in the route's ordered stop sequence.
3. Walk forward through the sequence: at each stop, add the average observed travel time to the next stop (`t_avg_to_next_min`, precomputed from the real data for that route). Emit a synthetic observation at that stop and timestamp.
4. Keep going until the gap end is reached, or `MAX_LOOPS=3` full route loops have elapsed.
5. When the bus hits the terminal stop (last stop in sequence), it wraps around to stop 0. The wraparound travel time is also computed empirically from observed data (average time from the last stop back to the first stop).
6. Tag rows `source = "imputed_forward"`.

**Backward fill** (used for `after_only` buses):
1. Take the bus's first known stop after the gap.
2. Walk *backward* through the stop sequence from that stop, subtracting travel times to arrive at earlier synthetic timestamps.
3. Stop once the timestamps reach or precede the gap start.
4. Tag rows `source = "imputed_backward"`.

**Fallback:** If a bus's anchor stop isn't found in the route stop sequence (e.g., because the stop match was ambiguous), `get_nearest_stop` is called as a fallback — it does a spatial search to find the closest stop in the sequence and uses that as the starting point.

**Result on fake 2h gap:** 4,176 rows, 100 routes, 528 vehicles.

**Limitation:** Accuracy degrades for long gaps — a bus walking forward for 5+ hours will drift from reality as schedules vary. Also depends on having reliable `t_avg_to_next_min` estimates, which require enough observed data per route.

---

### 3. `impute_gap_gtfs` — GTFS Schedule Fill

**The idea:** Ignore observed bus positions entirely. The GTFS static feed tells us which trips were *scheduled* to run and when. Use that schedule as a proxy for what was actually happening.

**How it works:**
1. Parse all `departure_time` strings from `stop_times.txt` into full timestamps on the gap date. (GTFS times can exceed 24:00:00 for overnight trips, so this requires careful handling.)
2. Filter to only the trips and stops whose scheduled departure falls within the gap window.
3. Subsample to approximately one observation every `target_freq_min=10` minutes per trip — to match the density of the real data.
4. Assign synthetic vehicle IDs in the format `gtfs_{route}_{trip_id}` (no real `vid` is known).
5. Tag rows `source = "imputed_gtfs"`.

**Result on fake 2h gap:** 2,419 rows, 43 routes, 778 vehicles.

**Limitation:** GTFS represents the *schedule*, not what actually ran. Buses that were cancelled, delayed, or running off-schedule won't match. Route coverage is lower because not all routes have scheduled trips in every time window. Also produces no real vehicle IDs.

---

## Supporting Functions

| Function | Purpose |
|---|---|
| `match_bus_to_stops` | Matches each GPS observation to its nearest valid stop for that route using a `cKDTree` spatial index. Valid stops are those actually served by the route (via `stop_times.txt`). Match radius = 0.0002 degrees. Matched 37.2% of the 8.2M rows. Adds a `stop_id` column — required for the fwd/bwd method. |
| `detect_gaps` | Scans the full sorted timestamp series and returns all windows > 10 minutes with no observations from any vehicle. Returns a DataFrame of `(gap_start, gap_end, duration_minutes)`. |
| `identify_missing_buses` | Given a gap window, classifies each vehicle as `both_sides`, `before_only`, or `after_only` based on whether it appears in a 2-hour lookback/lookahead window on each side of the gap. Returns three DataFrames (one per category) containing each bus's last/first known stop. |
| `create_fake_gap` | Removes a random time window of a specified length from the middle 80% of the data. Returns the gapped DataFrame, the held-out "truth" rows, and the gap start/end. Used for evaluation. |
| `compare_imputation_methods` | Given ground truth and imputation results from all three methods, plots stop-sequence-position vs. time trajectories for a specified route and generates coverage bar charts. For evaluation only (requires ground truth). |
| `plot_imputation` | Single-method version of the above for the real gap where no ground truth exists. |

---

## Evaluation

A synthetic 2-hour gap was created with `create_fake_gap` and all three methods were run against it. Results were compared against the held-out ground truth using `compare_imputation_methods`.

| Method | Rows imputed | Routes covered | Vehicles |
|---|---|---|---|
| fwd/bwd | 4,176 | 100 | 528 |
| fill | 4,896 | 37 | 186 |
| gtfs | 2,419 | 43 | 778 |

**Decision:** `fill` was chosen for the real gap despite lower vehicle count in the fake-gap test. On the real 333-minute gap, `fill` covered 116/116 active routes — substantially better than both alternatives on that longer window. The week-prior data was clean and available, making `fill` the most reliable option.

---

## Output

```
all_data_imputed_with_stops.csv
```

The imputed rows from `impute_gap_fill` (tagged `source = "imputed_same_period"`) appended to the original data. All original columns are preserved.
