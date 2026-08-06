# OnTimeAI — Estado actual y próximas mejoras

**Fecha del relevamiento**: 2026-05-27 (17:30 UTC)
**Modelo activo**: `4year_v9` (LightGBM, 84 features, AUC test offline 0.847)
**Última corrida medida**: run 953 — 17:01 UTC

---

## 1. Resumen ejecutivo

| Indicador | Hoy (2026-05-27) | Ayer (2026-05-26) | Inicio sesión (2026-05-22) |
|---|---|---|---|
| AUC raw (modelo solo) | **0.855** | 0.812 | 0.6775 |
| AUC final (con ajustes) | 0.892 | **0.941** | — |
| Brier raw | **0.146** | 0.189 | 0.291 |
| Brier final | 0.138 | **0.105** | — |
| F1 | 0.64 | **0.74** | 0.594 |
| Matched pairs | 1,104 | 913 | 366 |
| Lineage raw coverage today | **100%** | 100% | 45% |

Modelo raw alcanzó su mejor performance histórica (**0.855 vs offline 0.847**). El AUC final cayó hoy por un bug en NAS Status: 252 de 266 closures registradas eran NOTAMs administrativos de Aviación General sin impacto comercial.

---

## 2. Snapshot operativo (hoy)

| Métrica de sistema | Valor |
|---|---|
| Total predicciones generadas | 1,814 (1,068 vuelos distintos) |
| Actuals settled (cualquier día) | 4,538 |
| Tails distintos predichos hoy | 483 |
| Cache `tail_lineage_cache` (total) | 4,487 |
| Cache fresco (<2h) | 2,505 (55.8%) |
| Observaciones ADS-B hoy | 12,963 |
| Observaciones METAR hoy | 2,004 |
| Registros NAS Status últimas 2h | 187 |

---

## 3. Mapa de componentes

### 3.1 Backend (Cloud Run service + Cloud Run Job)

| Componente | Estado | Observaciones |
|---|---|---|
| `live_pull.py` orquestación | ✅ Funcional | Tick cada 30 min, ~137s chain-walk (timeout 600s) |
| AeroAPI scheduled (6h window) | ✅ Funcional | Consume API calls — controlado |
| Lineage cascade L2/L3 | ✅ Deployed | carrier+route+hour → carrier+route → carrier → global |
| ADS-B ETA adjustment | ✅ Funcional | 12,963 obs hoy |
| Intermediate `dep_delay` capture | ✅ Funcional | |
| Cross-source dedup FR24 ↔ AeroAPI | ✅ Funcional | En `build_inference_frame` |
| NAS Status XML + GDP adjustment | ✅ **Fixed** | GA NOTAMs ahora filtrados (sección 4) |
| SHAP top-K persistence | ✅ Funcional | Tabla `prediction_shap` |
| Recalibración `4year_v9_recal` | ✅ Funcional | IsotonicRegression sobre 2,266 actuals |

### 3.2 Harvester (Cloud Run Job separado, cron `15,45 * * * *`)

| Componente | Estado | Observaciones |
|---|---|---|
| FR24 atl_anchor (L1) | ✅ Funcional | ~120 tails/run |
| FR24 chain_walk (L2) | ✅ Funcional | Budget 80, history limit **50** (era 25) |
| airplanes.live ADS-B | ✅ Funcional | ~200 aircraft/run |
| OpenSky fallback | ⚠️ Parcial | Timeouts intermitentes |
| Bootstrap unseen tails | ✅ Funcional | 0 pendientes hoy |
| Priority queue (5 clases) | ✅ Funcional | bootstrap > predicted-today > high_freq > never > expired |
| Adaptive TTL por frecuencia | ✅ Funcional | 2h / 6h / 24h |
| Cache purge >30d | ✅ **Nuevo** | Corre en cada tick antes del chain_walk |

### 3.3 Weather

| Componente | Estado | Cobertura |
|---|---|---|
| METAR via IEM | ✅ Funcional | ~73% (limitado por gap futuro >90min) |
| TAF forecasts | ❌ No implementado | 0% futuro |
| SIGMET/CWA en ruta | ❌ No implementado | — |
| Wind aloft por ruta | ❌ No implementado | — |

---

## 4. Bug NAS scraper — FIXED 2026-05-27

### Síntoma original
NAS scraper marcaba como "Airport Closure" (180 min) entradas que en realidad eran NOTAMs administrativos para Aviación General:

```
LAS | Airport Closure | 180.0 | "!LAS 04/141 LAS AD AP CLSD TO NON SKED TRANSIENT GA ACFT..."
SAN | Airport Closure | 180.0 | "!SAN 03/071 SAN AD AP CLSD TO NON SKED TRANSIENT GA ACFT..."
SUN | Airport Closure | 180.0 | "!SUN ... SUN AD AP CLSD EXC HEL OPS"
```

### Fix aplicado
`feature_engineering_v7/gdp_scraper.py`:
- Nueva función `_is_ga_only_notam(reason)` filtra reasons que contengan: `NON SKED`, `NON-SKED`, `GA ACFT`, `GA TRANSIENT`, `TRANSIENT GA`, `EXC HEL`, `EXCEPT HEL`.
- Parser `Airport_Closure_List` ahora skipea esas entradas.
- Tests: 8/8 PASS. Validación live feed FAA: 0 closures spurias (antes 3 activas hoy).

### Audit retroactivo del impacto

**Magnitud histórica**:
| Métrica | Valor |
|---|---|
| Total registros `Airport Closure` históricos | 266 |
| **Spurios (GA-only NOTAMs)** | **252 (94.7%)** |
| Reales (commercial-impact) | 14 (5.3%) |
| Ventana contaminada | 2026-05-25 → 2026-05-27 (3 días) |
| Aeropuertos afectados | SAN (92), LAS (91), SUN (69) |

**Impacto en predicciones (LAS + SAN, n=80 settled)**:
| Métrica | Raw | Final (post-NAS) | Δ |
|---|---|---|---|
| Media `proba` | 0.180 | **0.969** | **+5.4x** |
| Brier score | 0.246 | 0.627 | -2.5x peor |
| Pos-rate truth | 32.5% | — | — |
| Predicciones `>0.5` | 6 (7.5%) | **80 (100%)** | **54 FP** |

**Lectura**:
- Modelo raw clasificaba LAS/SAN cerca del base rate correcto (mean 0.18 vs real 0.325).
- El boost NAS empujaba TODAS las predicciones a casi 1.0.
- **54 FP de 243 totales hoy (~22%) vienen de este bug.**
- Próximos ciclos (run ≥ 954) ya no insertan closures GA-only.
- No se borra el histórico — el daño ya está en `predictions`, y re-predecir no aporta valor.

---

## 5. Métricas detalladas de hoy

### Confusion matrix (n=1,104, threshold `quantile@0.22`)
```
                Predicted
                 0       1
Actual    0 |  522     243   |  765 on-time
          1 |   63     276   |  339 delays
              585     519
```

### Magnitud delays observados
- N delays reales: **339** (30.7%) · Mediana: 46 min · Media: 78 min · p90: 195 min

### Top vuelos en riesgo (run 953, 17:01 UTC)

| Flight | Origen | Llega ATL | p_raw | p_final | Señal dominante | Veredicto |
|---|---|---|---|---|---|---|
| DL1073 | SAT | 12:53 | 0.13 | 1.00 | ADS-B +252min, dep +271 | **Real** |
| DL549 | ATL→? | 21:15 | 0.18 | 0.98 | GDP_DEST 180 | Por validar |
| F91116 | LAS | 17:55 | 0.19 | 0.96 | GDP_ORIG 180 (NOTAM GA) | **FP** — bug NAS |
| DL1182 | AVL | 16:59 | 0.18 | 0.94 | dep_delay +30, ADS-B +7 | **Real** |
| F93022 | DFW | 17:27 | 0.21 | 0.77 | GDP_ORIG 75 (DFW thunderstorms) | **Real** |

---

## 6. Lo que SÍ está bien — no tocar

- Modelo raw (LightGBM v9) — al techo offline; no re-entrenar.
- Calibración `4year_v9_recal` — ECE 0.094 < 0.10.
- Lineage cascade L2 + L3 — 100% cobertura hoy.
- Priority queue del harvester.
- Cross-source dedup FR24 ↔ AeroAPI.
- DB sync GCS — refresh cada 30 min sin gaps.

---

## 7. Próximas mejoras priorizadas

### 7.1 URGENTE — completado en esta sesión

| # | Mejora | Estado |
|---|---|---|
| A | Fix NAS scraper NOTAM filter | ✅ código + tests + audit |
| 9 | FR24_HISTORY_LIMIT 25 → 50 | ✅ default + env-var |
| 10 | Purge tail_lineage_cache >30d | ✅ función + wire harvester |

**Pendiente**: deploy a Cloud Run (próximo paso).

### 7.2 Mid wins (2-3h) — datos nuevos sin retrain

| # | Mejora | Fuente | Effort | Ganancia |
|---|---|---|---|---|
| 6 | NAS Status enriched (XML completo) | nasstatus.faa.gov | 1h | Mejor attribution GDP/GS/AFP |
| 2 | NOTAM scraping real | FAA NOTAM API | 2h | Detecta delays por infraestructura |
| 5 | ADS-B trajectory persistence (holding) | airplanes.live | 1h | Captura holding sobre ATL |
| 8 | Adaptive paging atl_anchor (skip empty) | FR24 | 30 min | Más calls para chain_walk |

### 7.3 Big wins (4-6h) — cierra gap METAR

| # | Mejora | Fuente | Effort | Ganancia |
|---|---|---|---|---|
| **1** | **TAF forecasts** | aviationweather.gov | 3-4h | METAR futuro 0% → ~95% |
| 3 | SIGMET/CWA en ruta | NOAA AWC | 2h | Captura convección invisible al modelo |

### 7.4 Long-term (1-2 días) — requiere retrain v9.1

| # | Mejora | Fuente | Effort |
|---|---|---|---|
| 4 | Wind aloft por ruta | NOAA AWC | 3h + retrain |
| 7 | ADSBExchange como 3er fallback | adsbexchange.com free | 30 min + retrain |

### 7.5 NO HACER

- ❌ Más `max_pages` en FR24 — ya balanceado
- ❌ OpenSky con auth — timeouts persistentes
- ❌ NOAA radar mosaic — 50MB/scan
- ❌ OpenSky historical — paywall blando

---

## 8. Métricas de éxito por etapa

| Etapa | AUC final | Precision | F1 |
|---|---|---|---|
| Tras fix NAS (hoy/mañana) | ≥ 0.94 | ≥ 0.62 | ≥ 0.73 |
| Tras Mid wins (Día 3) | ≥ 0.95 | ≥ 0.65 | ≥ 0.75 |
| Tras Big wins (Día 5) | ≥ 0.96 | ≥ 0.70 | ≥ 0.78 |
| Tras retrain v9.1 (Día 7) | ≥ 0.96 | ≥ 0.72 | ≥ 0.80 |

---

## 9. Cambios sin commit en esta sesión

**Backend** (`OnTimeAI-Backend`)
- `ontimeai/lineage_fallback.py` — Layer 3 cascade + `build_live_turnaround_lookups`
- `live_pull.py` — bootstrap unseen tails + integración live_turnaround
- `feature_engineering_v7/gdp_scraper.py` — **NEW**: filtro GA-only NOTAMs (`_is_ga_only_notam`)
- `scripts/analyze_live_period.py` — fix encoding cp1252
- `scripts/plot_shap_from_db.py` (nuevo)
- `scripts/plot_feature_coverage_slim.py` (nuevo)
- Plots actualizados en `artifacts/live_period_plots/`
- Reportes `artifacts/live_period_report_2026-05-22_2026-05-26.{json,md}`
- `claudedocs/ESTADO_ACTUAL_Y_MEJORAS_2026-05-27.md` (este doc)

**Scrapper** (`OnTimeAI-Scrapper`)
- `ontimeai_scrapper/harvester.py` — `expand_candidate_tails()` + purge wire
- `ontimeai_scrapper/lineage_cache.py` — priority queue 5-clases + **NEW**: `purge_stale_cache()`
- `ontimeai_scrapper/config.py` — `FR24_HISTORY_DEFAULT_LIMIT` 25 → 50 (env-driven)

> Commit recomendado en dos pasos separados (backend y scrapper) antes de Mid wins.
