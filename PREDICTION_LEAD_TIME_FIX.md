# Fix — Predecir antes de la salida (lead time)

**Fecha**: 2026-06-01
**Repos involucrados**: OnTimeAI-Backend (este) + OnTimeAI-Scrapper (descubrimiento).

---

## 1. Síntoma

Pregunta original: *¿los vuelos se predicen antes o después de su salida?*

Midiendo `predicted_at_utc` vs `scheduled_out_utc` sobre 2 días de predicciones live
(timestamps verificados en UTC):

| Cohorte | Resultado |
|---|---|
| **Salidas de ATL** (origin=ATL) | 61.9% con ≥1 predicción antes de gate; **38.1% recién después**. Lead **mediana +4 min**, p90 sólo +24 min. **Mediana de 1 predicción por vuelo.** |
| **Arribos a ATL** (dest=ATL) | 95.7% predichos después de su salida de origen — **correcto por diseño** (se predice delay de arribo en vuelo). |

No es leak (nunca hay predicción post-aterrizaje), pero el lead time es de minutos, no útil.

### Falsa pista descartada
Un análisis por día mostraba un "cliff" el ~2026-05-12 (de 110 min de lead a ~5 min). Era
artefacto: las predicciones tempranas venían de **replay/backfill** (`scripts/replay_period.py`,
239 ticks/día con `predicted_at` sintético), no de ticks live. El comportamiento live (cron
limpio cada 30 min, ~48 ticks/día desde ~05-19) **siempre** tuvo el lead pobre. No hubo
regresión de un commit.

---

## 2. Causa raíz (dos defectos apilados)

1. **Target acoplado al fetch.** `live_pull.py:289` hacía
   `target_ids = sched_rows + arr_sched_rows` = sólo los vuelos traídos *en ese tick*. Un
   vuelo se predecía únicamente en el ciclo en que se descubría → mediana 1 predicción/vuelo.

2. **Horizonte de descubrimiento ~75 min.** Los *airport boards* de AeroAPI
   (`scheduled_departures`) **y** FR24 (`get_airport_details`) sólo exponen vuelos ~1h antes
   de salir. Evidencia: en el upload del harvester de las 01:47, la salida de ATL más futura
   era 02:50 (~75 min), pese a 124k vuelos FR24 ingeridos. La ventana de 6h en `live_pull.py`
   nunca se llenaba.

---

## 3. Fixes aplicados (este repo)

### Fix 1 — `live_pull.py`: target desacoplado del fetch
Reemplaza `target_ids = sched_rows + arr_sched_rows` por una query a la DB que toma **todas
las salidas de ATL aún por despegar** dentro del horizonte:

```sql
SELECT f.fa_flight_id
FROM flights f
LEFT JOIN actuals a ON a.fa_flight_id = f.fa_flight_id
WHERE f.origin = 'ATL'
  AND f.scheduled_out_utc IS NOT NULL
  AND a.actual_off_utc IS NULL                                              -- no despegó
  AND datetime(COALESCE(f.estimated_out_utc, f.scheduled_out_utc)) > :now   -- aún por delante (delay-aware)
  AND datetime(f.scheduled_out_utc) <= datetime(:now, '+:H hours')          -- dentro del horizonte
  AND COALESCE(f.cancelled, 0) = 0
```
…unido a los vuelos recién fetcheados. Horizonte por `PREDICT_HORIZON_HOURS`
(default = `--schedule-hours` = 6).

- El predicado `COALESCE(estimated_out_utc, scheduled_out_utc) > now` es **delay-aware**:
  mantiene en target a un vuelo demorado cuyo `estimated_out` se corre al futuro, y **excluye**
  los ~125/ciclo que ya despegaron pero no se settlearon (lag de actuals) — sin esto se
  generarían predicciones post-departure espurias.
- Costo ~cero: inferencia local, features ya en la DB, sin llamadas extra a AeroAPI.
- Efecto: una predicción pre-departure fresca por vuelo, **re-predicha cada 30 min** hasta el
  pushback (refina con weather/lineage nuevos).

### Fix 3 — `api.py` `_latest_predictions_today`: contrato pre-departure al servir
El `ROW_NUMBER()` ahora ordena **prefiriendo predicciones pre-departure**:

```sql
ROW_NUMBER() OVER (
    PARTITION BY p2.fa_flight_id
    ORDER BY
        CASE WHEN a2.actual_off_utc IS NOT NULL
                  AND p2.predicted_at_utc > a2.actual_off_utc
             THEN 1 ELSE 0 END ASC,   -- 0 = pre-departure (preferida)
        p2.predicted_at_utc DESC
) AS rn
```
Las salidas muestran su última predicción pre-departure; los arribos (que sólo tienen
predicciones en vuelo) caen a la última pre-landing en vez de quedar en blanco. Complementa
el commit `20a675b` (que excluía sólo predicciones post-*landing*).

**Validación**: `py_compile` OK; queries corridas contra la DB de prod (target=23 delay-aware
vs 150 naïf con 125 falsos positivos; serving=103 vuelos sin error); `pytest tests/test_api.py
tests/test_live_db.py` en verde (usar `-p no:debugging` por un bug local de `pyreadline`/3.13).

---

## 4. Pendiente: lead time real (el descubrimiento)

Fix 1 garantiza predicción *antes de salir* para todo vuelo que esté en la DB con tiempo, pero
el **lead sigue acotado por el descubrimiento (~75 min)**. Eso se resuelve en el
**OnTimeAI-Scrapper**, capturando las *future-legs* del chain-walk (el itinerario de cada tail
ya trae su próxima salida de ATL 1–3h+ adelante; hoy se descarta porque FR24 las deja con
`id==null`). Flag `CAPTURE_FUTURE_LEGS`. Ver `OnTimeAI-Scrapper/FUTURE_LEG_CAPTURE_DESIGN.md`.

**Dry-run validado (2026-06-01)**: 30 tails → 31 future-legs ATL, **lead mediana 521 min
(~8.7h)**, 100% >4h. Gate (≥120 min) **PASS**.

### Para activarlo (los dos juntos)
1. Harvester job: `CAPTURE_FUTURE_LEGS=true`
2. Backend job: **`PREDICT_HORIZON_HOURS=12`** — si no, las legs a 6–12h quedan descubiertas
   pero no se predicen hasta entrar a 6h (medido: 1 dentro de 6h vs 12 dentro de 12h).

---

## 5. Archivos tocados (este repo)
- `live_pull.py` — target desacoplado + predicado delay-aware (Fix 1).
- `api.py` — `_latest_predictions_today` prefiere pre-departure (Fix 3).
