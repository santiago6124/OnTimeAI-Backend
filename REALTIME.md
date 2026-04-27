# Real-Time Mode (Fase 2)

Pipeline de inferencia continua sobre datos en vivo de FlightAware AeroAPI + IEM METAR.

## Componentes

| Archivo | Rol |
|---|---|
| `ontimeai/live.py` | Cliente AeroAPI, schema SQLite, IEM puller, feature builder con history buffer |
| `live_backfill.py` | One-time: siembra `flights` + `actuals` con los últimos N días del master 2025 |
| `live_pull.py` | Per-tick: pull schedules + actuals + weather, predict, persistencia en SQLite |
| `live_metrics.py` | Métricas rolling sobre predicciones que ya tienen actuals |
| `live_data.db` | SQLite con `flights`, `predictions`, `actuals`, `weather_obs`, `runs` |

## Setup

1. **API key**: agregá `AEROAPI_KEY=...` en `.env` (gitignored).
2. **Backfill inicial**: `python3 live_backfill.py --days 14`
3. **Generar distance lookup** (si no existe): se construyó en `artifacts/distance_lookup.csv`.

## Cron diario

Ejemplo `crontab -e` (cada 30 min entre 06:00 y 23:00 UTC):

```
*/30 6-23 * * * cd /path/to/OnTimeAI-Backend && /usr/bin/python3 live_pull.py >> /var/log/ontimeai/live.log 2>&1
0 1 * * *      cd /path/to/OnTimeAI-Backend && /usr/bin/python3 live_metrics.py --since 24h --out artifacts/live_metrics_$(date +\%Y\%m\%d).json
```

## CLI flags

`live_pull.py`:
- `--schedule-hours N` (default 2): pull scheduled departures for next N hours
- `--actuals-hours N` (default 4): pull arrived flights from past N hours to settle actuals
- `--max-pages N` (default 2): client-side cursor pagination cap
- `--skip-arrivals-sched` / `--skip-actuals` / `--no-weather`: cost controls
- `--dry-run`: print plan, no API calls

`live_metrics.py`:
- `--since 24h|7d|2w` (default: all-time)
- `--threshold-min` (default 15)
- `--out PATH`: dump JSON

## Costo y rate limits

Cada tick puede consumir, en orden de magnitud:

| Endpoint | Calls per tick | Notas |
|---|---|---|
| scheduled_departures | 1-2 | ~15 vuelos/page |
| scheduled_arrivals | 1-2 | mismo |
| arrivals (actuals) | 1-2 | mismo |
| IEM METAR (16 estaciones) | 16 | gratis, separado de AeroAPI |

48 ticks/día × 4-6 calls AeroAPI = **~250 calls/día**, ~7,500/mes.

El cliente hace **backoff exponencial** ante 429 (8s, 16s, 32s) y respeta cursor pagination con sleeps de 2s.

## Esquema SQLite

```sql
flights (fa_flight_id PK, ident_iata, op_carrier, flight_number, tail_num,
         origin, dest, inbound_fa_flight_id, fl_date, crs_dep_min,
         scheduled_*_utc, crs_elapsed_min, distance, aircraft_type,
         cancelled, diverted, first_seen_utc, last_updated_utc)

predictions (fa_flight_id, predicted_at_utc PK, proba_delay, predicted_delay)

actuals (fa_flight_id PK, actual_*_utc, arr_delay_min, departure_delay_min,
         cancelled, diverted, settled_at_utc)

weather_obs (station, valid_utc PK, tmpc, dwpc, relh, drct, sknt, alti,
             p01m, vsby, gust, wxcodes, *_flag)

runs (run_id PK, started_utc, finished_utc, flights_pulled, flights_predicted,
      actuals_updated, weather_obs_added, notes)
```

## Limitaciones conocidas

- **Lineage requiere ≥ 7 días de history live**: el backfill desde 2025 sirve solo
  como warmup; las features `prev_arr_delay_tail` solo trabajan bien una vez
  acumulado el history live (~24-48 h después del primer cron).
- **Carriers internacionales**: vuelos con `op_carrier` no presente en BTS
  (ej: BA, KE, KL, AF, JL) se mappean a NaN en categorical lookup. La predicción
  igual corre, pero el modelo nunca vio esa categoría.
- **TAIL_NUM**: AeroAPI lo devuelve consistentemente; AviationStack free no.
- **Rate limit**: AeroAPI Personal Free tier tiene quotas estrictas. Con cron
  cada 30 min y `--max-pages 2`, vamos justos. Subir a Personal $10/mo si
  querés más resolución.

## Estado actual

- ✅ Pipeline operacional, predicciones reales escritas a SQLite
- ✅ Schema completo con runs/observability
- ⏳ Necesita 24-48 h de cron continuo para acumular history → métricas live confiables
- ⏳ FastAPI endpoint `/predict` y dashboard Next.js: separados, en `OnTimeAI-Frontend`
