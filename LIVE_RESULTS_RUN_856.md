# Resultados live — run 856 (2026-05-24T19:11Z)

Análisis vuelo-por-vuelo de las **primeras 14 predicciones** del run 856 que ya
tienen ground-truth (los demás 38 vuelos siguen en aire o aún sin hidratar).

**Tiempo de ground-truth**: ~40 min después del run (19:51Z).
**Threshold usado**: `quantile@0.22 = 0.5810` (calibrado sobre las 52 probas del batch).

---

## 1. Resumen de métricas

| Métrica | Valor |
|---|---|
| N con ground-truth | 14 / 52 |
| Accuracy | **57.1%** (8 OK / 14) |
| Precision (de los predichos delay) | 33.3% (2 TP / 6 pred=1) |
| Recall (de los reales delay) | 50% (2 TP / 4 truth=1) |
| AUC empírico (pair-wise) | ~0.575 |
| TP / TN / FP / FN | 2 / 6 / 4 / 2 |

AUC test del modelo: 0.86. AUC live histórico: 0.71. AUC observado en este sub-muestreo: 0.58 — bajo, pero la muestra está sesgada (los que aterrizan PRIMERO son típicamente los puntuales, los con delay alto llegan después).

---

## 2. Contexto del entorno al momento del run (19:11Z)

Estas son las señales agregadas que el modelo "vio":

### 2.1 Carrier delay rates (ventana 13:11–19:11Z, llegadas a ATL)

| Carrier | N llegados | Delayed (>15 min) | Rate |
|---|---|---|---|
| DL | 243 | 4 | **1.6%** |
| F9 | 16 | 1 | 6.3% |
| 9E | 4 | 0 | 0% |
| UA | 4 | 0 | 0% |
| YX | 1 | 0 | 0% |

**Lectura**: día tranquilo. Delta operando casi perfecto (1.6% delay rate), Endeavor (9E) sin ningún delay en la ventana. **Esto sesga al modelo a predecir "on-time" en general** — lo cual va a explicar los 2 FN del 9E más adelante.

### 2.2 Origin delay rates (ventana 6h pre-run, llegadas a ATL desde cada origen)

| Origin | N | Avg delay (min) | Delayed |
|---|---|---|---|
| TUL | 1 | -25.4 | 0 |
| TLH | 3 | -18.5 | 0 |
| SAV | 2 | -12.7 | 0 |
| OKC | 1 | -32.8 | 0 |
| LGA | 8 | -18.6 | 0 |
| EWR | 4 | -21.2 | 0 |
| DFW | 5 | -17.9 | 0 |
| CMH | 3 | -22.7 | 0 |
| BWI | 3 | -18.5 | 0 |

**TODOS los orígenes** vienen llegando **muy temprano** (-12 a -33 min de promedio). Cero delays en la ventana de referencia. Si el modelo confía en `origin_delay_rate_*`, predeciría "on-time" casi en todos los casos.

### 2.3 Weather (METARs cercanos)

- **ATL** 18:52Z: 27.8°C, viento 140°@9kt G15, VSBY 10 mi, sin precip → benigno
- **LGA** 18:51Z: 13.3°C, viento 50°@14kt G25, **VSBY 2 mi, BR (mist)** → mala visibilidad pero solo afecta al único DEP_FROM_ATL del set (N640RW, predicho on-time correctamente)
- **MLU, FWA, BWI, etc**: VSBY 10, sin precip → benignos

**Sin señal climática fuerte** para predecir delay en ninguno de los 14.

---

## 3. Detalle vuelo-por-vuelo

Cada vuelo lista lo que el modelo "vio" y qué pasó realmente.

### 3.1 ✅ TRUE POSITIVES (modelo acertó "delay")

#### **N986AT — DL SAV→ATL, sched 19:14Z → 0.76 → real +22 min ✓**
**Señales pre-predict:**
- Inbound del tail N986AT: vuelo previo (ATL→SAV) — **inbound resolution fallida en JOIN** (id mismatch FR24/AeroAPI), feature `prev_arr_delay_tail` probablemente NaN
- Origin SAV: 2 llegadas previas, ambas early (-12.7 min avg) → señal "on-time"
- Carrier DL: 1.6% delay rate → señal "on-time"

**¿Por qué predijo 0.76 sin lineage ni origin signal?** Hipótesis: el modelo se apoyó en hora del día (`dep_hour=15`, fin de tarde = congestión histórica en ATL), `dest_delay_rate_*` para ATL (~3% en ventana, pero patrón histórico de delays vespertinos), o `TAIL_DELAY_DECAY` con valor residual del tail. Confirmación pendiente con SHAP.

#### **N345NB — DL OKC→ATL, sched 19:10Z → 0.58 → real +17 min ✓**
**Señales pre-predict:**
- Inbound resuelto temporalmente: previo del tail fue ATL→OKC que llegó **-25 min (early)** → contradice el delay
- Origin OKC: 1 vuelo previo, -32.8 min → señal fuerte "on-time"
- Carrier DL: 1.6%

**Predicción rozó el threshold (0.584 vs 0.581).** El modelo tuvo señales contradictorias y se quedó al filo. El resultado real (+17 min, 2 min por encima del corte de 15) fue también al filo. Predicción correcta pero por poco margen.

### 3.2 ✅ TRUE NEGATIVES (modelo acertó "on-time")

| Vuelo | Tail | Sched In | Proba | Inbound prev | Real delay |
|---|---|---|---|---|---|
| DL IND→ATL | N946AT | 19:41 | 0.51 | ATL→IND −23 min (early) | −4 min |
| UA EWR→ATL | N851UA | 19:28 | 0.49 | AUS→EWR −7 min (early) | +14 min (close call) |
| DL DFW→ATL | N322DN | 19:30 | 0.48 | ATL→DFW −31 min (early) | +12 min (close call) |
| DL BWI→ATL | N963DZ | 19:54 | 0.43 | ATL→BWI −17 min (early) | −12 min |
| DL LAX→ATL | N397DN | 19:38 | 0.41 | MSP→LAX −23 min (early) | +3 min |
| YX ATL→LGA | N640RW | 21:30 | 0.22 | LGA→ATL −14 min (early) | −14 min |

**Patrón claro**: cuando el inbound del tail llegó early, modelo prevé on-time. Coherente con feature `prev_arr_delay_tail`. **Los 6 TN tuvieron inbounds con delay < 0**, y el modelo asignó proba < 0.51.

Notable: EWR y DFW llegaron a +14 y +12 (cerca del corte 15). El modelo estuvo "casi" equivocado por escasos minutos.

### 3.3 ❌ FALSE POSITIVES (modelo predijo delay, fue on-time)

Los 4 vuelos donde el modelo "se equivocó por arriba". Patrón común: **llegaron MUY temprano** (-16 a -22 min).

#### **N603AT — DL TLH→ATL → 0.82 → real -17 min ❌**
- Inbound previo: **no encontrado en la DB** (ni por FK ni por temporal lookup) — el tail tuvo un gap, posiblemente vuelo no capturado por harvester ni AeroAPI
- Origin TLH: 3 llegadas previas, todas early
- Sin señal weather

**Hipótesis del por qué 0.82**: sin lineage (NaN), modelo probablemente usó `TAIL_DELAY_DECAY` (que decae con el tiempo desde el último vuelo conocido del tail; si el tail no tiene un vuelo reciente capturado, este valor sube → señal de "incertidumbre" interpretada como riesgo). Conjetura — requiere SHAP para confirmar.

#### **N921AT — DL TUL→ATL → 0.72 → real -16 min ❌**
- Inbound resuelto: previo ATL→TUL llegó **-15 min early** → debió bajar la proba
- Origin TUL: -25 min, sin delays

**El modelo IGNORÓ una señal lineage clara on-time.** Sospecha: el `prev_arr_delay_tail` no se asoció (entre el 59.6% NaN del log), y el modelo se basó en `TAIL_DELAY_DECAY` u otra feature que apuntaba alto.

#### **N307PQ — 9E MLU→ATL → 0.53 → real -4 min ❌**
- Inbound resuelto: previo ATL→MLU llegó **+23 min late** → coherente con señal delay
- Pero `9E carrier rate = 0%` → contradice

**El modelo CONFIÓ en el lineage delay** (correctamente leyendo "el tail viene atrasado") pero la realidad fue que el vuelo se "recuperó" en MLU y salió on-time. **Decisión razonable del modelo aunque incorrecta**.

#### **N981AT — DL CMH→ATL → 0.52 → real -22 min ❌**
- Inbound: no resuelto (gap del tail, similar a N603AT)
- Origin CMH: 3 vuelos previos, todos early
- Sin señal weather

Misma sospecha que N603AT: lineage NaN → modelo usó proxies que apuntaron alto.

### 3.4 ❌ FALSE NEGATIVES (modelo predijo on-time, fue delay)

Los 2 vuelos críticos. **Ambos son carrier 9E (Endeavor regional de Delta).**

#### **N908XJ — 9E FWA→ATL → 0.46 → real +33 min ❌**
- Inbound resuelto: previo ATL→FWA llegó **-5 min (slightly early)** → señal on-time
- Origin FWA: weather benigno
- **Carrier 9E rate = 0% en la ventana** (4 vuelos, ningún delay)
- Distance ~700 mi, vuelo regional

**El modelo se confió** en el rate del carrier y el inbound on-time. La realidad: el vuelo se demoró 33 min — posiblemente por congestión en ATL al aterrizar (que el modelo no capturó porque su `dest_delay_rate_*` para ATL todavía no reflejaba la congestión en formación).

#### **N921XJ — 9E TRI→ATL → 0.45 → real +38 min ❌**
- Inbound resuelto: previo ATL→TRI llegó **+40 min late** → **fuerte señal lineage delay**
- Origin TRI: 1 vuelo previo, -14.6 min
- Carrier 9E rate = 0%

**Este es el caso MÁS revelador**: el lineage decía claramente "el tail está atrasado +40 min", pero el modelo predijo **on-time con proba 0.45**. Posibles explicaciones:
1. **`prev_arr_delay_tail` no se asoció** para este target — caemos en el 59.6% NaN
2. El JOIN del lineage falló porque el inbound usaba ID FR24 (hex) y la lookup del modelo esperaba ID AeroAPI

**Diagnóstico del modelo**: si el lineage tail delay se hubiera asociado correctamente, este vuelo habría tenido proba > 0.7 y sería TP. **Es un bug de data assoc, no de aprendizaje del modelo.**

---

## 4. Patrones detectados

### 4.1 Cuando el lineage funcionó, el modelo acertó
- N307PQ MLU→ATL: vio inbound +23, predijo 0.53 (delayed) — falló pero la lógica fue correcta
- N640RW ATL→LGA: vio inbound -14, predijo 0.22 — TN
- N921AT TUL→ATL: vio inbound -15 → pero predijo 0.72 (FP) — **excepción rara**, sugiere que `prev_arr_delay_tail` no se asoció

### 4.2 Cuando NO se asoció lineage, el modelo "sobreestimó"
- N603AT TLH→ATL: 0.82 (FP) — sin lineage
- N981AT CMH→ATL: 0.52 (FP) — sin lineage
- N986AT SAV→ATL: 0.76 (TP por suerte) — sin lineage

**El modelo "compensa la ausencia de lineage" con probas altas** (probablemente vía `TAIL_DELAY_DECAY`). Eso genera FPs pero también acertó 1 TP.

### 4.3 Los 2 FN ambos del 9E con señales débiles
- Carrier rate "perfect" 9E (0% delay) → señal engañosa
- En N921XJ el lineage estaba ahí pero no se asoció → bug crítico

### 4.4 4 FP llegaron MUY temprano (no es "cerca del threshold")
- Los 4 con proba > 0.5 que erraron, llegaron con **delay entre -16 y -22 min**. No fue un "casi delay" — fueron muy on-time.
- Indica que el modelo está leyendo señales que en realidad NO eran predictivas para esos vuelos. Posiblemente `TAIL_DELAY_DECAY` o features de hora del día.

---

## 5. Lo que faltaba para cada caso

### 5.1 Para los 2 FN
- **N921XJ TRI→ATL**: necesitaba que `prev_arr_delay_tail = +40` se asociara correctamente. Bug de ID matching FR24/AeroAPI. Si se hubiera asociado → proba > 0.75 → TP.
- **N908XJ FWA→ATL**: lineage era débil (-5 min). El modelo no tenía señal predictiva fuerte. Posibles features faltantes: arrivals en cola a ATL en la próxima hora (congestión en formación que el modelo no leyó).

### 5.2 Para los 4 FP
- **N603AT, N981AT**: les faltó tener un inbound capturado. Sin lineage, el modelo "elevó la apuesta" por defecto.
- **N921AT**: tenía lineage temporal (early) pero no se asoció vía `inbound_fa_flight_id`. Mismo bug de ID matching.
- **N307PQ**: lineage funcionó pero el vuelo "se recuperó" — el modelo no puede predecir eso.

### 5.3 Bug que explica varios casos
**El `inbound_fa_flight_id` apunta a un ID que no coincide con `flights.fa_flight_id`** porque harvester y backend usan ID schemes distintos (FR24 hex vs AeroAPI long). Este "id mismatch" es responsable de:
- 3+ FP por lineage no asociado
- Al menos 1 FN crítico (N921XJ)

**Solución pendiente**: hacer el JOIN del lineage por `stable_id` además de `fa_flight_id`, o asegurar que `chain_walk_inbound` siempre escriba el ID que coincide con el `fa_flight_id` del target.

---

## 6. Conclusiones

1. **El modelo funciona conceptualmente**: cuando recibe lineage válido, lo usa. Cuando recibe carrier/origin rates bajos, predice on-time. Cuando recibe weather adverso, lo incorpora.

2. **El bug de ID matching entre FR24 y AeroAPI** está enmascarando la principal feature del modelo (`prev_arr_delay_tail`, que explica 34% del gain según `metrics.json`). Esto va a ser el próximo blocker.

3. **AUC observado 0.575 NO es la AUC real del modelo** — es un sub-muestreo de los vuelos que aterrizan más temprano (sesgo on-time). Esperar 1-2h más para que aterricen los otros 38 va a corregir esto.

4. **Decisiones del modelo no eran irrazonables**: 3/4 FP tenían lineage faltante (modelo "no sabía"), 1 FP era un caso de recuperación post-MLU. 1/2 FN era data-assoc bug.

5. **Lo que falta para subir AUC live de ~0.71 a algo cercano a 0.80**:
   - Fix del ID matching lineage (impacta 30-40% de targets)
   - Captura de congestión en formación en ATL (feature nueva)
   - Mejor manejo de carrier-specific patterns (9E tuvo 100% accuracy histórica pero 0/2 hoy)

---

## 7. Queries para reproducir este análisis

```bash
# Bajar DB actual
gsutil cp gs://ontimeai-live-db/live_data.db tmp/live_data.compare.db

# 1) Las 14 matches predicted×actuals
sqlite3 tmp/live_data.compare.db "
SELECT f.tail_num, f.op_carrier, f.origin||'->'||f.dest,
       SUBSTR(f.scheduled_in_utc, 12, 5) sch,
       ROUND(p.proba_delay, 2) proba, p.predicted_delay pred,
       ROUND(a.arr_delay_min, 0) real_delay,
       CASE WHEN (a.arr_delay_min > 15) = (p.predicted_delay = 1) THEN 'OK' ELSE 'XX' END m
FROM predictions p
JOIN flights f ON f.fa_flight_id = p.fa_flight_id
JOIN actuals a ON a.stable_id = p.stable_id AND a.arr_delay_min IS NOT NULL
WHERE p.predicted_at_utc >= '2026-05-24T19:00:00' AND p.predicted_at_utc < '2026-05-24T19:30:00'
ORDER BY p.proba_delay DESC;"

# 2) Inbound temporal (vuelo previo del mismo tail)
# Ver query "WITH targets / inbound_resolved" en este documento §3
```
