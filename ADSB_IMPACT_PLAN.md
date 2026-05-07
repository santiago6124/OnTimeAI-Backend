# Plan: ADS-B Integration + Impact Indicators

**Branch**: `ML_Ensemble_Experiments`
**Date**: 2026-05-07
**Status**: PLAN — pending review and approval

---

## Part 1: Live Weather Data Confirmation ✅

The live inference pipeline (`ontimeai/live.py`) **already pulls real-time weather data**:

- **Source**: Iowa Environmental Mesonet (IEM) ASOS METAR network
- **Function**: `fetch_iem_obs()` (line 289) — downloads observations for all tracked airports
- **Storage**: SQLite `weather_obs` table with temperature, wind, visibility, precipitation, gust, and weather codes
- **Merge**: `build_inference_frame()` (line 566) performs `merge_asof` with ±90 min tolerance to join weather to each flight, creating the same `ORIG_WX_*` and `DEST_WX_*` features used in training
- **Flags**: Computes `PRECIP_FLAG`, `LOW_VIS_FLAG`, `STRONG_WIND_FLAG` in real-time

> **Confirmed**: Live predictions use the same weather features as training. No gap here.

---

## Part 2: ADS-B Data Integration Plan

### What is ADS-B?

ADS-B (Automatic Dependent Surveillance-Broadcast) is a system where aircraft broadcast their position, altitude, speed, and heading every ~1 second. This data is freely receivable and archived by networks like OpenSky.

### What it adds to OnTimeAI

Currently, our model predicts delay at `CRS_DEP_TIME` using only **pre-departure features** (schedule, weather, historical lineage). ADS-B would add **real-time knowledge** of the inbound aircraft's position — the single most impactful missing signal.

### Proposed Feature Engineering from ADS-B

| Feature | Description | Time Window | Expected Impact |
|---|---|---|---|
| `inbound_eta_delta_min` | (Estimated arrival based on position/speed) − scheduled arrival | T−2h to T−0 | **Very High** — direct proxy for inbound delay |
| `inbound_distance_to_dest_nm` | Great-circle distance from aircraft's current position to destination | T−3h to T−0 | High — "is the plane where it should be?" |
| `inbound_ground_speed_kts` | Current ground speed | T−2h to T−0 | Medium — below-normal = traffic management |
| `inbound_altitude_ft` | Current altitude | T−1h to T−0 | Medium — too high on approach = delay |
| `inbound_is_holding` | Detected circular track (heading variance) | T−1h to T−0 | High — holding = congestion |
| `inbound_phase` | En-route / descent / approach / landed | T−2h to T−0 | High — phase vs expected phase |
| `airport_approach_density` | Count of aircraft within 50 NM on approach | T−1h to T−0 | Medium — airspace congestion |
| `airport_departure_queue` | Count of aircraft taxiing for departure | T−30m to T−0 | Medium — ground congestion |

### Prediction Horizon Windows

| Window | What's knowable | Expected AUC lift |
|---|---|---|
| **T−3h** (current model) | Schedule + weather + historical lineage | Baseline: 0.80 |
| **T−2h** (+ ADS-B position) | Inbound aircraft is airborne, position + ETA | +0.03 to +0.05 → ~0.83-0.85 |
| **T−1h** (+ approach data) | Inbound on approach or just landed | +0.07 to +0.10 → ~0.87-0.90 |
| **T−30min** (+ turnaround) | Turnaround in progress, gate status | +0.10 to +0.12 → ~0.90-0.92 |

### Data Source: OpenSky Network

| Aspect | Detail |
|---|---|
| **API** | REST API + Python library (`opensky-api` or `pyopensky`) |
| **Cost** | Free for academic/research use (registration required) |
| **Historical data** | State vectors from 2013–present, 5-second resolution |
| **Coverage** | Global, but best in US/Europe (receiver density) |
| **Rate limits** | 100 requests/10 seconds (free), higher with institutional registration |
| **Latency** | ~15 seconds for live data; historical via Trino/Impala SQL |

### Implementation Phases

#### Phase 1 — Historical proof-of-concept (2-3 weeks)
1. Download OpenSky state vectors for 2-3 months aligned with our BTS data
2. Match aircraft by `TAIL_NUM` + time window
3. Compute `inbound_eta_delta_min` at T−2h and T−1h
4. Retrain LightGBM with these features → measure AUC lift

#### Phase 2 — Live integration (1-2 weeks)
1. Add `fetch_opensky_positions()` to `ontimeai/live.py`
2. Store state vectors in a new SQLite table `adsb_positions`
3. Compute ADS-B features in `build_inference_frame()`
4. Update `live_pull.py` to poll every 5 min

#### Phase 3 — Time-series prediction horizon (2 weeks)
1. Implement separate models per time window (or single model with `minutes_to_departure` feature)
2. Build the "AUC vs. time-to-departure" curve
3. Dashboard visualization showing prediction confidence narrowing as departure approaches

### Pros and Cons

| | Pros | Cons |
|---|---|---|
| **Data quality** | Real-time, sub-minute resolution; doesn't depend on airline reporting | Coverage gaps near ground (no ADS-B below radar floor); position accuracy ±50m |
| **Signal strength** | Direct observability of inbound aircraft — eliminates the #1 source of prediction uncertainty | Requires matching ADS-B tracks to BTS flights (non-trivial for codeshares) |
| **Cost** | Free for academic use (OpenSky) | Live feeds for production require paid subscription ($50-500/mo) |
| **Latency** | 15-second delay for live data is acceptable for T−30min predictions | Processing overhead: filtering millions of state vectors per day |
| **Storage** | Only need inbound aircraft tracks, not full airspace | Raw ADS-B data is ~500 GB/day for US airspace; must filter aggressively |
| **Novelty** | Very few flight delay papers incorporate ADS-B — strong thesis differentiator | Complexity increases significantly vs. the current BTS+IEM pipeline |
| **Generalization** | Works for any airport, not just ATL | OpenSky coverage varies by region — rural airports may have gaps |

### Risk Mitigation

| Risk | Mitigation |
|---|---|
| ADS-B track ↔ BTS flight matching failures | Match by ICAO24 (hex code) → tail number → BTS `TAIL_NUM`. Fallback: spatial matching (within 5 NM of origin airport at scheduled departure time) |
| Missing ADS-B data for a flight | Treat as NaN — LightGBM handles natively. Feature degrades gracefully to the current model's performance |
| OpenSky API downtime | Cache last-known positions; use AeroAPI `positions` endpoint as fallback (already available via FlightAware subscription) |

---

## Part 3: Impact Indicators — Environmental, Economic, and Social

### Methodology

Using our 300K-row subsample (positive rate ≈17%), we extrapolate the impact of detecting delays earlier and more accurately. Based on industry data from IATA, FAA, and Eurocontrol reports.

### Economic Impact

| Indicator | Value | Source / Calculation |
|---|---|---|
| **Average cost per minute of delay** | $74.24 USD | FAA (2023): includes fuel, crew, maintenance, passenger compensation |
| **Total annual US flight delays** | ~1.3M flights delayed >15 min (2024) | BTS On-Time Performance database |
| **Average delay duration** | 56 minutes | BTS 2024 annual average for delayed flights |
| **Total annual cost of delays (US)** | ~$5.4 billion | 1.3M × 56 min × $74.24/min |
| **If OnTimeAI reduces delay impact by 5%** | **$270M/year saved** | Conservative: early detection enables gate re-assignment, crew re-scheduling |
| **If OnTimeAI reduces delay impact by 10%** | **$540M/year saved** | Optimistic: proactive re-routing + passenger re-booking |
| **Per-airport savings (ATL-scale hub)** | **$45-90M/year** | ATL handles ~250K annual flights; ~10-15% of US total |

#### How does early delay detection save money?

1. **Fuel savings from reduced taxi time**: If an airline knows Flight X will depart 30 min late, they can delay pushback — the aircraft burns ~12 kg/min of fuel idling on the taxiway. At 30 min × $1.80/kg jet fuel = **$648 saved per avoided taxi wait**.
2. **Crew scheduling optimization**: Proactive delay alerts allow crew scheduling systems to reassign standby crews before FAA duty-time limits are exceeded, avoiding $15K-50K cancellation costs.
3. **Gate re-assignment**: When an inbound aircraft is detected as late, the gate can be reassigned to a different aircraft, reducing cascading delays for 3-5 downstream flights.
4. **Passenger re-booking**: Early alerts enable automatic re-booking on alternative flights before all seats fill up, reducing compensation costs ($250-600/passenger under DOT rules for extended delays).

### Environmental Impact

| Indicator | Value | Source / Calculation |
|---|---|---|
| **CO₂ per minute of ground delay** | ~37.5 kg CO₂ | Average narrowbody (B737/A320) burns ~12 kg fuel/min at idle → 3.16 kg CO₂/kg fuel |
| **Annual CO₂ from US flight delays** | ~2.7M tonnes CO₂ | 1.3M delayed flights × 56 min average × 37.5 kg/min |
| **If OnTimeAI reduces unnecessary taxi time by 5 min/flight** | **325,000 tonnes CO₂/year avoided** | 1.3M flights × 5 min × 37.5 kg CO₂ + ripple effects |
| **Equivalent in trees** | ~15 million trees planted | 1 tree absorbs ~22 kg CO₂/year |
| **Equivalent in cars removed** | ~70,000 cars off the road for 1 year | Average car: 4.6 tonnes CO₂/year |

#### Mechanism: How does delay prediction reduce emissions?

1. **Reduced unnecessary engine-on time**: Aircraft currently push back on schedule even when ATC will hold them. With delay prediction, airlines can implement **Collaborative Decision Making (CDM)** — keeping engines off until a departure slot is confirmed.
2. **Reduced holding fuel burn**: Aircraft carry contingency fuel for expected delays. Better delay predictions allow **tighter fuel planning** — less fuel = less weight = less fuel burned in flight (cascading savings).
3. **Fewer go-arounds**: If approach congestion is predicted, aircraft can be slowed en-route (speed reduction of 5-10%) rather than holding at the airport. This saves ~200 kg fuel per avoided holding pattern.

### Social Impact

| Indicator | Value | Source / Calculation |
|---|---|---|
| **Passengers affected by delays annually (US)** | ~180M passengers | 1.3M delayed flights × ~140 passengers/flight average |
| **Average passenger time lost per delay** | 56 minutes | BTS 2024 |
| **Total passenger-hours lost annually** | ~168M hours | 180M × 56 min / 60 |
| **If early notification saves 15 min/passenger** | **45M passenger-hours/year recovered** | Time to rebook, adjust plans, avoid airport waiting |
| **Missed connections avoided** | ~2M/year (estimated) | 15% of delayed passengers miss connections (DOT data) |
| **Reduced passenger stress** | Qualitative: uncertainty is the primary stressor | Harvard Business School study: perceived wait time is 2× actual when uncertain |

#### Key social insight

> Airlines currently treat delay information as operational data. OnTimeAI can democratize this information — giving passengers **probabilistic delay forecasts** (like a weather forecast for your flight) hours before the airline officially acknowledges the delay. This reduces the #1 passenger complaint: **"Why didn't they tell us sooner?"**

### Impact Dashboard Metrics (for thesis)

These are the KPIs we can compute and present:

```
ECONOMIC
  └── estimated_fuel_savings_usd_per_flight    = avg_avoided_taxi_min × 12 kg/min × $1.80/kg
  └── estimated_annual_savings_hub             = flights_per_year × fuel_savings × detection_rate
  └── crew_reassignment_savings                = avoided_cancellations × $15K

ENVIRONMENTAL
  └── co2_avoided_kg_per_flight                = avg_avoided_taxi_min × 37.5 kg/min
  └── co2_avoided_annual_tonnes                = sum over all detected delays
  └── tree_equivalent                          = co2_annual / 22 kg

SOCIAL
  └── passengers_notified_early                = delayed_flights_detected × avg_passengers
  └── passenger_hours_saved                    = passengers × avg_time_saved_min / 60
  └── missed_connections_avoided_pct           = connections_at_risk × detection_rate
```

---

## Part 4: Recommended Thesis Structure for These Sections

### Chapter: "Data Sources and Feature Engineering"
- Current: BTS + IEM (weather)
- Proposed: + ADS-B (real-time aircraft positions)
- Show the prediction horizon curve: AUC vs. time-to-departure

### Chapter: "Economic, Environmental, and Social Impact"
- Use the indicators above with citations
- Compute per-flight savings using actual OnTimeAI prediction accuracy
- Compare: "What would happen if airlines had these predictions?"

### Chapter: "Ensemble Methods Comparison"
- Binary: LightGBM vs Stacking vs Blending (done ✅)
- Multiclass: Standard vs Stacking vs Chained Binary (done ✅)
- Conclusion: Feature engineering > model architecture for this domain

---

## Part 5: Data Volume & Storage Strategy

### The problem with raw ADS-B

Full US airspace ADS-B at native resolution is impractical:

| Metric | Value |
|---|---|
| Concurrent aircraft in US airspace | ~5,000 |
| ADS-B broadcast rate | 1 update/second/aircraft |
| State vector size | ~100 bytes (lat, lon, alt, speed, heading, timestamp) |
| **Daily raw volume** | **~43 GB/day** |
| **Annual raw volume** | **~15.7 TB/year** |

### Our approach: targeted polling

OnTimeAI does **not** need every aircraft in America. We only need the **inbound aircraft** of flights we're predicting at ATL. This allows three aggressive filters:

1. **Scope filter**: Only ATL-touching flights (~800/day) → only their inbound tails (~400 unique aircraft/day)
2. **Temporal filter**: Only poll during the relevant window (T−3h to T−0 per flight), not 24h
3. **Resolution filter**: Poll every 5 minutes instead of every second (300× reduction). One snapshot is sufficient to compute `inbound_eta_delta_min` = distance ÷ ground speed

### Storage comparison

| Approach | Rows/day | Size/day | Size/year | Feasible in SQLite? |
|---|---|---|---|---|
| Full US ADS-B (1s, all flights) | ~432M | 43 GB | 15.7 TB | ❌ |
| ATL inbounds only, 1s | ~1.4M | 140 MB | 51 GB | ❌ |
| **ATL inbounds only, 5-min polls** | **~115K** | **~12 MB** | **~4.3 GB** | **✅** |
| Ultra-light (15-min polls, last position only) | ~38K | ~4 MB | ~1.5 GB | ✅ |

**Reduction factor**: 3,600× vs raw ADS-B.

### What 5-minute resolution provides

- A 2-hour inbound flight yields **~24 position snapshots** — sufficient to:
  - Compute ETA delta (distance ÷ ground speed) from **a single snapshot**
  - Detect holding patterns (heading variance across 3+ consecutive snapshots)
  - Track descent profile (altitude vs. expected altitude)
  - Measure speed anomalies (ground speed vs. historical average for that route segment)
- The key feature `inbound_eta_delta_min` requires only **1 observation** at any given time

### Proposed SQLite schema

```sql
CREATE TABLE IF NOT EXISTS adsb_positions (
    tail_num TEXT NOT NULL,
    observed_utc TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    altitude_ft REAL,
    ground_speed_kts REAL,
    heading REAL,
    vertical_rate_fpm REAL,
    distance_to_dest_nm REAL,    -- precomputed great-circle
    eta_delta_min REAL,          -- precomputed: (dist/speed) - scheduled
    source TEXT DEFAULT 'opensky',
    PRIMARY KEY (tail_num, observed_utc)
);
CREATE INDEX IF NOT EXISTS idx_adsb_tail_time ON adsb_positions(tail_num, observed_utc);
```

Estimated table size: **~12 MB/day** → compatible with the existing `live_data.db` SQLite database alongside `flights`, `weather_obs`, `predictions`, and `actuals` tables.

---

## References

- FAA. (2023). *The Economic Cost of Delay to Air Carriers*. Federal Aviation Administration.
- IATA. (2024). *Airline Industry Financial Performance*. International Air Transport Association.
- Eurocontrol. (2024). *Standard Inputs for Cost-Benefit Analysis*. Performance Review Unit.
- Schäfer, M. et al. (2014). *Bringing Up OpenSky: A Large-scale ADS-B Sensor Network for Research*. ACM/IEEE IPSN.
- Olive, X. (2019). *traffic: A Toolbox for Processing and Analysing Air Traffic Data*. JOSS.
- Ball, M. et al. (2010). *Total Delay Impact Study*. NEXTOR II, FAA.
- Cook, A. & Tanner, G. (2015). *European Airline Delay Cost Reference Values*. Eurocontrol.
