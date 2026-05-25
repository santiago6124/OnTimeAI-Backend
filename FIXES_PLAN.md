# Plan de fixes para mejorar accuracy en live tests

**Fecha**: 2026-05-24
**Base diagnóstica**: `LIVE_RESULTS_RUN_856.md`, `HALLAZGOS_LIVE_PREDICT_FIX.md`, `PLAN_HARVESTER_LINEAGE.md`, `artifacts/data_quality_report.json`
**AUC live actual**: ~0.71 (vs 0.86 test) — gap de 14 pts
**Objetivo**: AUC live ≥ 0.78 sin re-entrenar el modelo (cerrar el gap por integridad de datos en runtime, no por capacidad del modelo)

---

## Diagnóstico revisado (corrige `LIVE_RESULTS_RUN_856.md §5.3`)

El doc previo atribuyó las fallas de lineage a "ID mismatch FR24 hex vs AeroAPI long en `inbound_fa_flight_id`". Tras leer `ontimeai/lineage.py:30-128`, **la feature `prev_arr_delay_tail` NO se calcula por JOIN en `inbound_fa_flight_id`** — se calcula sobre el DataFrame concatenado (history + target) sorted por `(TAIL_NUM, EVENT_ORIGIN_UTC)`. Para cada vuelo, toma el delay del leg previo del mismo tail que arrived antes de su `scheduled_off_utc`.

Las **causas reales** del 59.6% NaN observado en run 856 son tres, en orden de impacto:

### Causa 1 — JOIN cross-source falla
`build_inference_frame` (`ontimeai/live.py:593-601`) hace:
```sql
SELECT f.*, a.arr_delay_min FROM flights f JOIN actuals a ON a.stable_id = f.stable_id
```
- **Harvester FR24** (`OnTimeAI-Scrapper/ontimeai_scrapper/fr24_client.py:274,330`): escribe `stable_id = fa_flight_id` (hex completo, sin normalizar)
- **AeroAPI** (`ontimeai/live.py:185-190`): `stable_id()` strippea sufijo → `"IDENT-TIMESTAMP"`

Para el **mismo vuelo físico**, los dos sources crean filas con stable_id distintos. La JOIN falla cross-source: si harvester escribió `flights[stable_id=hex]` pero solo AeroAPI capturó actuals con `stable_id=IDENT-TS`, el vuelo queda sin `arr_delay` en history → no contribuye al cómputo de lineage del siguiente leg del tail.

### Causa 2 — `DISABLE_LINEAGE_FALLBACK=1` quedó activo en Cloud Run
Per `HALLAZGOS_LIVE_PREDICT_FIX.md §6`, este env var se agregó como diagnóstico durante el hunt del SIGSEGV. **El SIGSEGV era CRLF, no el fallback.** Sin embargo el env var siguió activo y `live_pull.py:255` lo respeta — el modelo recibe NaN duros donde debería tener priors poblacionales (`ontimeai/lineage_fallback.py:185-200`).

Sin fallback, el modelo extrapola con `TAIL_DELAY_DECAY` u otras features proxy y sobreestima (los 4 FP de run 856 vinieron sin lineage).

### Causa 3 — `chain_walk_max=20` con 52 targets = 38% cobertura
`live_pull.py:62`. Con presupuesto AeroAPI de 20 calls/tick, solo los primeros 20 targets en orden de iteración (no ordenados) reciben chain-walk para hidratar su inbound. Los otros 32 quedan sin lineage si el harvester tampoco los tenía.

---

## Plan priorizado

### Tier 1 — Lo que puedo hacer ahora sin tocar producción

| # | Fix | Esfuerzo | Impacto AUC esperado | Requiere |
|---|---|---|---|---|
| **A** | Re-enable lineage_fallback (default False en env, en vez de bypass activo) | 5 min | +0.02 a +0.04 inmediato (recupera priors poblacionales) | Deploy job |
| **B** | Cross-source actuals reconciliation pos-JOIN por (TAIL_NUM, scheduled_off_utc) | 1-2h | +0.03 a +0.05 (recupera lineage perdida cross-source) | Deploy job |
| **C** | Priorizar chain_walk por proximidad temporal (`scheduled_in` ASC) | 30 min | +0.01 a +0.02 (mejor uso del presupuesto AeroAPI) | Deploy job |
| **D** | Persistir SHAP top-15 por predicción en tabla nueva | 1h | 0 directo, **bloquea futura iteración** sin esto | Deploy job + backend |
| **E** | `.gitattributes` + pre-commit anti-CRLF en `*.lgb` | 10 min | 0 (defensivo) | Nada |
| **F** | Diagnosticar TZ regression run 857 (solo investigación) | 1h | TBD según hallazgo | Nada |

### Tier 2 — Requiere decisión del usuario

| # | Fix | Razón |
|---|---|---|
| G | Recalibrar `4year_v9_recal` con >12K actuals acumulados | Cambia artifact en producción |
| H | Race condition arquitectónico (bucket separado para predictions) | Decisión arquitectónica + nuevo bucket GCS |
| I | Feature `atl_arrivals_in_window_30min` (congestión) | Requiere re-entrenar el modelo |
| J | Carrier rate con Bayesian smoothing | Cambia distribución de features → necesita retrain |
| K | NAS/FAA GDP feature | Nueva fuente + retrain |

### Tier 3 — Roadmap post-MVP

| # | Fix | Razón |
|---|---|---|
| L | `ontimeai/cascade.py` LSTM/GNN per docstring | Post-MVP per código mismo |
| M | Switch a Cloud SQL | Costo +$10-30/mes, último recurso |

---

## Plan de ejecución autónomo (esta sesión)

Orden secuencial. Cada fix se commitea solo cuando el siguiente está validado contra el código que comparte el área.

1. ✅ Plan documentado (este archivo)
2. ⏳ **Fix A** — Re-enable lineage_fallback (default-on)
3. ⏳ **Fix B** — Cross-source reconciliation
4. ⏳ **Fix C** — Chain-walk priorización temporal
5. ⏳ **Fix E** — `.gitattributes` CRLF defense
6. ⏳ **Fix F** — Diagnóstico TZ regression (read-only)
7. ⏳ **Fix D** — SHAP persistence (después de A/B/C porque depende del shape del feature pipeline)

Lo que NO voy a hacer sin instrucción:
- Ejecutar `./deploy.sh`
- `gcloud run jobs execute ontimeai-live-pull`
- Commits a git
- Quitar el env var `DISABLE_LINEAGE_FALLBACK` en Cloud Run (requiere `gcloud run jobs update`)
- Modificar `4year_v9*` artifacts (recalibración)

---

## Validación esperada

Tras aplicar A+B+C+E y deployar, en el próximo run:

| Métrica | Antes (run 856) | Esperado tras fixes |
|---|---|---|
| `lineage_hit_rate` log message | ~40% | **≥75%** |
| Predicciones con `prev_arr_delay_tail` NaN | 59.6% | **<25%** |
| Mean target proba | ~0.39 | ~0.30 (sin sobreestimar por NaN) |
| AUC observado (cuando aterricen) | ~0.58 (n=14) | **≥0.72** |

---

## Comandos de validación post-deploy

```bash
gsutil cp gs://ontimeai-live-db/live_data.db tmp/live_data.postfix.db
sqlite3 tmp/live_data.postfix.db "
  SELECT predicted_at_utc, COUNT(*) n_pred,
         AVG(proba_delay) avg_proba, AVG(predicted_delay) pos_rate
  FROM predictions
  WHERE predicted_at_utc >= '2026-05-24T20:00:00'
  GROUP BY substr(predicted_at_utc, 1, 13);"
```
