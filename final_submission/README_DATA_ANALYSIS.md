# Bus Performance Analysis — `data_analysis.ipynb`

End-to-end analysis of Chicago bus reliability using real-time AVL observations matched to GTFS schedule data. Computes a composite **Bus Trust Score** for every route, produces presentation-ready figures, and includes a deep-dive on Route 201.

---

## Data Inputs

| File | Description |
|---|---|
| `metrics_results/mar1_analysis.csv` | Raw stop-level observations from the AVL matcher (`match_type`, `delay_minutes`, `is_ghost`, etc.) |
| `metrics_results/performance_results.csv` | Pre-aggregated route-level performance metrics (OTP %, delay stats, ghost rate, TOD breakdowns) |
| `gtfs_cache/static_data.parquet` | Static GTFS schedule data used for the route map (stop coordinates, sequences, trip IDs) |
| `Datasets/ridership.csv` | Daily ridership counts by route, used for the Trust vs Ridership scatter |
| `metrics_results/mar1_full_results.csv` | Full results CSV used for the ridership merge figure |

---

## Setup

```bash
pip install pandas numpy matplotlib geopandas contextily pyarrow
```

The notebook uses a custom dark theme applied globally via `plt.rcParams`. All color constants (`GREEN`, `YELLOW`, `ORANGE`, `RED`, `BLUE`, `GREY`, `WHITE`) are defined in the setup cell and reused across every figure.

---

## Notebook Structure

### 1. Data Loading & Cleaning
Loads the two primary CSVs and filters observations to valid matched trips — excluding pre-trip signals, ghost pings, and rows with missing delay values. This mirrors the filter used inside `analyze_route_performance`.

### 2. Trip-Level Aggregation
Collapses stop-level observations into one row per trip, computing delay statistics (median, max, P90, std). Derives temporal features — hour, day of week, weekend flag, and time-of-day bucket — from the scheduled departure time.

**Time-of-day buckets:**
| Label | Hours |
|---|---|
| Early Morning | 00:00–05:59 |
| AM Peak | 06:00–09:59 |
| Midday | 10:00–14:59 |
| PM Peak | 15:00–18:59 |
| Evening | 19:00–23:59 |

### 3. Bus Trust Score
Computes a 0–100 composite score for each route from four normalized pillars:

| Pillar | Weight | Source column |
|---|---|---|
| Schedule Adherence (OTP) | 45% | `overall_otp_pct` |
| Delay Severity | 25% | `overall_delay_median` |
| Reliability | 20% | `overall_delay_std` |
| Ghost Penalty | 10% | `ghost_rate_pct` |

Routes are assigned a trust tier: **HIGH** (≥80), **MODERATE** (≥65), **LOW** (≥50), **CRITICAL** (<50).

---

## Figures Produced

| File | Description |
|---|---|
| `pres_1_trust_distribution.png` | Histogram of Bus Trust Scores across all 123 routes, shaded by tier |
| `pres_2_best&worst_routes.png` | Compact table — top 5 and bottom 5 routes |
| `pres_2a_best&worst_routes.png` | Full ranked table — all 123 routes |
| `report_route_map_chicago.png` | Geographic route map of the 4 worst routes overlaid on a Chicago basemap |
| `pres_3_worst_routes.png` | Pillar breakdown bar chart for the 4 worst-scoring routes |
| `pres_4_ghost_distribution.png` | Ghost trip rate ranked across all routes |
| `pres_5_ghost_types.png` | Pie chart of ghost signal types (G1 Freeze / G2 Jump / G3 Off-route / G4 Never departed) |
| `pres_6_weekday_median_delay_by_hour.png` | Weekday median trip delay by hour of day |
| `pres_7_weekday_pct_late_by_hour.png` | Weekday % of trips late (>3 min) by hour |
| `report_23h_spike_analysis.png` | Two-panel investigation of the 23:00 late-trip spike |
| `report_accumulated_delay.png` | Cumulative delay buildup throughout the service day |
| `report_1_summary_stats_table_transposed.png` | Delay summary statistics by time-of-day (transposed table) |
| `report_2_delay_by_tod.png` | Median delay + stacked OTP breakdown by time-of-day |
| `report_4_bottom3_stats.png` | Bottom 3 routes vs system average across 4 metrics |
| `report_5_otp_heatmap.png` | Route × TOD OTP heatmap — all routes, sorted worst to best |
| `pres_201_otp_donut.png` | Route 201 OTP breakdown donut chart |
| `pres_201_tod_boxplot.png` | Route 201 delay distribution by time-of-day |
| `pres_201_daily_tod_boxplot.png` | Route 201 day-level median delay by time-of-day |
| `pres_201_corridor_delay_profile.png` | Route 201 stop-level delay at Foster and Haven on Sheridan |
| `pres_11_trust_vs_ridership.png` | Bus Trust Score vs Ridership Score scatter — four-quadrant view |

---

## Route 201 Deep Dive

A focused section examines Route 201 at multiple levels of granularity:

- **Trip level** — OTP bucket breakdown, delay by time-of-day (boxplot and daily aggregation)
- **Stop level** — Northbound observations at Sheridan & Foster and Sheridan & Haven filtered to `des == 'Old Orchard'`, aggregated to daily medians with IQR spread

---

## Ridership Integration

Ridership data (`Datasets/ridership.csv`) is filtered to 2025 onwards, averaged by route, and min-max normalized to a 0–1 `ridership_score`. This score is merged with the trust metrics to produce the quadrant scatter (`pres_11`), identifying routes that are high-ridership but low-trust — the highest-priority intervention candidates.

---

## Key Definitions

- **On time** — median trip delay between -1 and +5 minutes
- **Late** — median trip delay > 5 minutes
- **Very late** — median trip delay > 15 minutes
- **Ghost trip** — vehicle broadcasting position but not completing its assigned trip
- **OTP %** — percentage of trips classified as on time or early
