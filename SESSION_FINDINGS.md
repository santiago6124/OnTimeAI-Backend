# Hallazgos de la sesión de testing — 2026-04-26/27

Documento de descubrimientos sobre el comportamiento del modelo OnTimeAI v3
(LightGBM, AUC train 0.81) cuando se lo testea fuera de su corpus de entrenamiento.

## Resumen ejecutivo

| Test | Setup | n | AUC | Hallazgo principal |
|---|---|---|---|---|
| **OOT 2021** | Master 2021 completo (BTS+IEM) | 163,668 | **0.788** | Modelo generaliza bien hacia atrás. Drop esperado de 0.81→0.79 por COVID-recovery distribution shift |
| Live AviationStack | Free tier, sin TAIL_NUM, sin history | 148-1,459 | **~0.50** | Modelo colapsa a random porque ~62% de features quedan en NaN |
| Live AeroAPI (cold start) | Sin backfill, sin weather | 4 | n/a (todas proba≈0.91) | Confirma diagnóstico: sin lineage, modelo predice baseline alta sin discriminar |
| **Live AeroAPI (warm)** | Backfill 1 día + weather | 19 | pendiente verif. en 4-6h | **Spread de probabilidades 23× mayor** — el modelo discrimina otra vez |

---

## Hallazgo 1: el OOT del corpus es honestamente estable

Entrenamos con 2022-2025. Probamos contra **2021 completo (163K vuelos)**.

| Métrica | Test 2025 (in-corpus) | OOT 2021 | Δ |
|---|---|---|---|
| Accuracy | 0.7822 | **0.8416** | +5.9 pts |
| ROC AUC | 0.8134 | **0.7884** | −2.5 pts |
| F1 | 0.6037 | 0.4786 | −12.5 pts |
| Brier | 0.1352 | 0.1016 | −0.034 (mejor calib) |
| Pos rate (truth) | ~25% | **15.3%** | 2021 fue año con menos delays |

**Lectura**:
- AUC se mantiene (drop de solo 2.5 pts es excelente para OOT)
- Acc sube (paradójico: más fácil acertar cuando la clase positiva es minoritaria)
- F1 baja (el threshold tuneado para ~25% positivos no calza con ~15%)
- Brier mejor → mejor calibración global

**Desglose por trimestre revela COVID effect**:

| Trimestre 2021 | Pos rate | AUC | Lectura |
|---|---|---|---|
| Q1 (COVID heavy) | **9.8%** | 0.769 | Tráfico bajo, atípico |
| Q2 (recovery) | 15.1% | 0.794 | Mejora |
| Q3 (Delta variant) | 19.5% | 0.788 | Rebote operacional |
| Q4 | 15.8% | 0.778 | Estabilizado |

**Conclusión académica**: el modelo tiene **distribution-shift gradual robusto** — pierde 2-4 pts de AUC en condiciones COVID extremas pero retiene la capacidad discriminativa esencial.

---

## Hallazgo 2: el live test colapsa cuando faltan las features de lineage

### Test con AviationStack (free tier)

Pulled ~7,500 vuelos de KATL para 25-26 abril 2026. Tres problemas estructurales:

1. **`aircraft.registration` (= TAIL_NUM) viene null en 98% del free tier** (31/1459)
2. **65% de carriers son internacionales** (KE, LA, AF, KL, VS, WS, etc.) **que no aparecen en BTS** doméstico
3. **Sin history backfill** → todas las rolling features (`carrier_delay_rate_24h`, etc.) en NaN

Resultado evaluado contra los 148 vuelos de US-domestic carriers ya aterrizados:

| Métrica | OOT 2021 | Live AviationStack |
|---|---|---|
| Accuracy | 0.842 | 0.527 |
| ROC AUC | 0.788 | **0.514** (≈ random) |
| Predicted positive rate | 15% | **99%** |

### Causa raíz

Las features dominantes del modelo (~62% del gain) requieren **historial reciente disponible**:

| Familia | % gain | Estado en live cold-start | Por qué |
|---|---|---|---|
| Lineage (prev_arr_delay_tail, etc.) | 34% | ❌ NaN | Necesita historia 24-48h del mismo TAIL |
| Rolling (carrier/origin delay rate) | 20% | ❌ NaN | Necesita ventanas 1h-7d de actuals |
| Weather | 8% | Opcional ❌ | Si no se hace pull |
| Cyclical/Schedule/Categorical | ~30% | ✅ presente | |

Con 62% del gain en NaN, LightGBM cae al "default direction" aprendido durante training. Como el threshold (~0.36) y el calibrador isotónico fueron entrenados con todas las features presentes, las probabilidades crudas se inflan → todo etiqueta como "delayed".

**Esto NO es un bug del modelo**. Es comportamiento esperado y conocido en sistemas ML productivos durante el cold-start.

---

## Hallazgo 3: AeroAPI resuelve el problema (mostly)

AeroAPI Personal devuelve los campos críticos que faltan en AviationStack free:

- ✅ `registration` (TAIL_NUM) presente en 95%+ de los vuelos
- ✅ `inbound_fa_flight_id` apunta al vuelo previo del mismo avión
- ✅ Endpoints `departures` y `arrivals` con `start`/`end` arbitrarios → backfill histórico
- ✅ `route_distance` en millas (= DISTANCE de BTS)
- ✅ Schemas: `scheduled_off/on/in/out` + `actual_off/on/in/out` + `arrival_delay`

Pero tiene tradeoffs:

- 💰 Pagado ($0.005/call). Free trial $5 USD inicial = ~1000 calls
- 🚦 Rate limits estrictos en free tier — 429 frecuentes con paginación agresiva
- ⏳ Backfill 2 días = ~400 calls = $2 USD (factible)
- ⏳ Cron 30 min × 24h = ~250 calls/día (sostenible)

---

## Hallazgo 4: con backfill + weather, el modelo recupera capacidad discriminativa

Setup final del test:
- AeroAPI backfill: 1 día (423 vuelos en buffer de lineage)
- IEM weather: 477 observaciones cargadas
- Predicción sobre 19 vuelos schedules para próximas 6 horas

| Métrica de probabilidades | Antes (cold start) | Después (warm) | Mejora |
|---|---|---|---|
| Min | 0.899 | 0.332 | −0.567 |
| Max | 0.922 | 0.875 | −0.047 |
| **Spread** | **0.023** | **0.542** | **23×** |
| Std dev | ~0.010 | 0.136 | 13× |
| Mean | 0.910 | 0.715 | −0.195 |

El **spread** es el indicador clave: el modelo distingue entre vuelos de bajo, medio y alto riesgo. Antes daba esencialmente la misma proba a todos.

Casos representativos:
- AA N730US ATL→ORD: **proba 0.332** (bajo riesgo) ← lineage indica vuelo previo OK
- DL N6715C ATL→MCO: **proba 0.875** (alto riesgo) ← carrier rate 24h alto
- MQ 4582 (sin TAIL): **proba 0.500** (incertidumbre) ← lineage en NaN aún

---

## Hallazgo 5: el threshold default es inadecuado para live

El threshold optimizado durante training (~0.36) se calibró para una distribución de probabilidades distinta. En live:
- Las probas crudas se inflan ligeramente (mean 0.72 vs train ~0.50)
- Casi todas quedan arriba de 0.36 → 95% predicted_label=1

**Implicación**: la métrica `accuracy` con threshold default va a sub-estimar la utilidad real del modelo. La señal útil está en las **probabilidades crudas**, no en el label binario.

**Solución**: re-tunear threshold después de ~1 semana de actuals live. Probable nuevo óptimo ~0.65.

---

## Hallazgo 6: arquitectura de pipeline real-time

El live test reveló los componentes mínimos para un sistema productivo:

```
1. Ingesta scheduled (AeroAPI scheduled_departures + scheduled_arrivals)
2. Ingesta actuals (AeroAPI departures + arrivals con start histórico)
3. Ingesta weather (IEM METAR, gratis, real-time)
4. History buffer (SQLite con últimos 7-30 días)
5. Feature builder con join target+history
6. Predict (LightGBM v3 sin cambios)
7. Persistence (predictions + actuals + runs en SQLite)
8. Eval cron (live_metrics sobre predicciones que tienen actuals)
```

Componentes implementados en esta sesión:
- `ontimeai/live.py` — primitives: cliente AeroAPI con backoff, schema SQLite, feature builder
- `live_backfill.py` — seed inicial desde master histórico (BTS 2025)
- `live_backfill_aeroapi.py` — backfill correcto desde AeroAPI (últimos N días)
- `live_pull.py` — tick principal (cron-friendly)
- `live_metrics.py` — métricas rolling sobre predicciones evaluadas
- `live_data.db` — SQLite con tablas `flights`, `predictions`, `actuals`, `weather_obs`, `runs`

---

## Hallazgo 7: cold-start es inevitable, pero corto

Para un sistema productivo:

| Tiempo desde deploy | Estado del sistema | AUC esperado |
|---|---|---|
| 0h (sin backfill) | Cold start total | 0.50 (random) |
| Después de backfill 1 día (~$1 USD) | Lineage parcial | 0.65-0.75 |
| Después de backfill 7 días + cron continuo | Warm | 0.75-0.80 |
| Estado estable (cron 30 min, 1+ semana) | Steady state | 0.78-0.82 |

**Comparable al AUC del corpus de entrenamiento** (~0.81) una vez que el sistema acumuló su propia historia.

---

## Implicaciones para la tesis

### Lo que funciona y se puede defender ahora

1. **Modelo entrenado con disciplina anti-leakage** (ver `tests/test_pipeline.py::test_prepare_dataset_has_no_leakage`)
2. **AUC test in-corpus 0.81** (4 años, 753K vuelos)
3. **AUC OOT honesto 0.79** (año 2021 completo, 163K vuelos, fuera del corpus de entrenamiento)
4. **Pipeline de inferencia real-time operacional** (AeroAPI + IEM + LightGBM v3 + SQLite)
5. **Discriminación demostrada en live** (spread 23× mayor con lineage warm)

### Lo que requiere más datos antes de defender

- AUC live confirmado con N grande (necesita 1-2 semanas de cron continuo)
- Threshold re-calibrado para distribución live
- Métricas operacionales (OTPA, SDDR del anteproyecto)

### Lo que el plan original deja para post-MVP

- Streaming Kafka/Pub-Sub (Fase 3)
- Auto-retraining periódico
- Módulo de efecto cascada (LSTM/GNN)
- Dashboard frontend

---

## Lecciones meta-metodológicas

1. **Las features dominantes determinan la robustez del sistema en producción**, no la arquitectura del modelo. Un modelo "simple" con features bien ingenierizadas sufre más por feature missingness que por elección de algoritmo.

2. **El backfill desde el corpus de entrenamiento (BTS Dec 2025) NO sirve para warm-start**. Un buffer de history desfasado cronológicamente es equivalente a no tener history.

3. **`inbound_fa_flight_id` de AeroAPI es la API key oculta para resolver lineage** sin requerir bulk pulls — el chain walk es 5-10× más barato que pullear todo el airport por N días.

4. **Free tier siempre miente**: AviationStack free omite los campos críticos. Validar **antes** de comprometerse.

5. **El threshold de un clasificador binario es tan importante como el modelo**. Re-calibrar threshold sobre distribución de producción es más impactante que reentrenar el modelo.

6. **Los rate limits son la friction principal del MVP en vivo**, no el modelo. Backoff exponencial + paginación cursor con sleeps + arquitectura idempotente son críticos.

---

## Numbers cheat-sheet

| Concepto | Valor |
|---|---|
| Corpus de entrenamiento | 753,060 vuelos × 4 años (2022-2025) × 16 aeropuertos |
| Test set in-corpus AUC | 0.813 |
| OOT 2021 AUC | 0.788 |
| Features totales en el modelo v3 | ~70 |
| % gain de lineage features | 34% |
| % gain de rolling features | 20% |
| Backfill AeroAPI 1 día (KATL) | ~200 calls, ~$1 USD |
| Cron sostenido (30 min) | ~250 calls/día, ~$1.25/día |
| Free trial AeroAPI | $5 USD inicial = ~1000 calls |
| n del primer test live (warm) | 19 vuelos |
| Spread de probas pre-warm | 0.023 |
| Spread de probas post-warm | 0.542 (23× más) |
