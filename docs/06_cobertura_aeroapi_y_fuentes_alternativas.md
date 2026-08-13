# Cobertura de AeroAPI y fuentes alternativas de datos de vuelo

**Fecha de análisis:** 2026-08-12  
**Contexto:** Investigación surgida al verificar el claim "+1.800 predicciones/día" en la UI.

---

## 1. Situación actual

### Volumen real predicho por día

| Métrica | Valor |
|---|---|
| Operaciones totales ATL/día (BTS) | ~2.700 |
| Vuelos únicos predichos/día (datos reales) | ~939 |
| Cobertura efectiva | ~35% |
| AUC live (`4year_v9_recal`) | 0.8197 |
| AUC en-route | 0.9245 |
| Actuals acumulados en DB | 13.741 |

El claim "+1.800 predicciones/día" que aparece en documentación es **incorrecto**. El número real es ~939 vuelos únicos diarios. La cifra de 1.800 coincide con el total teórico de operaciones comerciales ATL (900 salidas + 900 llegadas), que sería el alcance máximo del sistema — no el actual.

### Configuración de ingesta

```
MAX_PAGES=7  →  7 páginas × ~15 vuelos/página = 105 vuelos/endpoint/ciclo
Endpoints por ciclo: scheduled_departures + scheduled_arrivals + 2 actuals = 22 calls
Ciclos por día: 48 (cada 30 min)
Calls totales/día: ~1.056  |  ~31.680/mes
```

`MAX_PAGES=10` fue probado y saturó el rate limit de AeroAPI (registrado en CLAUDE.md, fix 2026-05-31).

---

## 2. Desalineamientos identificados

### 2.1 Features de delay rate calculadas sobre muestra parcial

Las features `origin_delay_rate_6h`, `dest_delay_rate_1h`, `carrier_delay_rate_24h` y similares se calculan usando únicamente los vuelos presentes en la DB. Con el 35% de cobertura, estas tasas son estimaciones sobre muestra incompleta — pueden subestimar o sobreestimar el delay real del aeropuerto en momentos de alta actividad.

Estas features son de las más influyentes según SHAP (aparecen consistentemente en el top-5 de contribuciones).

### 2.2 Tail lineage con tasa de missing del 71.6%

Las features de linaje de cola del avión son las más predictivas cuando están disponibles:

```
TAIL_DELAY_DECAY         → top feature, 71.6% missing
prev_arr_delay_tail      → top feature, 71.6% missing  
prev_turnaround_tail_min → top feature, 71.6% missing
```

El 71.6% de missing se debe a que el vuelo anterior del mismo avión no está en la DB — ese vuelo fue capturado por AeroAPI en otro momento o nunca entró al tope de paginación. LightGBM imputa los nulls internamente, pero se pierde la señal predictiva más fuerte del modelo.

### 2.3 Sesgo de selección en la evaluación del AUC

AeroAPI ordena los resultados por hora de salida (más próximos primero). Esto implica que **sistemáticamente capturamos vuelos con 1-2h de antelación** y perdemos los de 4-6h. El AUC se mide sobre el subconjunto que el sistema observa, no sobre el universo completo de vuelos ATL. El AUC real sobre todos los vuelos ATL podría diferir.

### 2.4 Impacto estimado

| Problema | Impacto estimado en AUC | Notas |
|---|---|---|
| Delay rates sobre muestra parcial | Moderado (~2-4 pp) | Afecta features top-5 |
| Tail missing 71.6% | Alto | La feature más predictiva falla en 7 de 10 casos |
| Sesgo de selección en evaluación | Bajo | El modelo se evalúa sobre lo que predice |

---

## 3. Fuentes alternativas investigadas

### 3.1 OpenSky Network ⭐ Recomendada para actuals

- **Costo:** Gratis para uso académico/investigación
- **Cobertura:** 100% de vuelos con transponder ADS-B (prácticamente todos los comerciales)
- **API:**
  ```
  GET https://opensky-network.org/api/flights/arrival?airport=KATL&begin=UNIX&end=UNIX
  GET https://opensky-network.org/api/flights/departure?airport=KATL&begin=UNIX&end=UNIX
  ```
- **Devuelve:** `firstSeen`, `lastSeen`, `callsign`, aeropuertos origen/destino
- **Limitación crítica:** Solo tiene tiempos reales (`actual_out`, `actual_in`) — **no da scheduled times con antelación**. No sirve para predecir antes del vuelo, pero sí para mejorar la cobertura de actuals usados en reentrenamiento.
- **Uso propuesto:** Reemplazar o complementar el endpoint de actuals de AeroAPI → más actuals → mejor AUC y reentrenamiento más robusto.

### 3.2 FAA ASWS (gratis, sin registro)

```
https://api.faa.gov/asws/json/airport/status/KATL
```

No devuelve vuelos individuales. Provee estado del aeropuerto, Ground Delay Programs activos y demoras promedio. Útil como feature de contexto operacional. El repo ya tiene un GDP scraper en `feature_engineering_v7/gdp_scraper.py` que cumple parte de este rol.

### 3.3 AviationStack

- **Costo:** $29/mes → 10.000 requests/mes
- **API:** `GET /v1/flights?dep_iata=ATL` con scheduled + actual times
- **Limitación:** No tiene `fa_flight_id` nativo de FlightAware — el match por número de vuelo + fecha requiere lógica adicional de deduplicación.
- **Evaluación:** No prioritario mientras AeroAPI esté activo.

### 3.4 Scraping de FlightAware website

Técnicamente factible (HTML parseable en `flightaware.com/live/airport/KATL`). Conflicto con ToS — zona gris en contexto académico. **No recomendado.**

### 3.5 AeroAPI con queries segmentadas por aerolínea ⭐ Mejor ROI

La mejora de mayor impacto sin costo adicional: en lugar de `scheduled_departures` global con MAX_PAGES=7, hacer queries por aerolínea:

```python
# Delta opera ~35% de ATL, American ~15%, United ~10%
airlines = ["DAL", "AAL", "UAL", "SWA", "SKW"]
for airline in airlines:
    fetch("/airports/KATL/scheduled_departures", airline=airline, max_pages=3)
```

Con 5 aerolíneas × 3 páginas = 15 páginas equivalentes de cobertura específica, pero dentro del rate limit porque las queries son más pequeñas y selectivas. Estimación: de ~939 a ~1.400-1.600 vuelos únicos/día.

---

## 4. Decisión

**No se implementará ninguna fuente adicional por el momento.** El equipo decidió enfocarse en otras prioridades para la tesis. Esta investigación queda registrada para:

- Documentar la limitación de cobertura como parte del análisis de sistema en la tesis
- Referencia futura si se decide escalar el sistema post-entrega
- Justificar por qué el AUC reportado es sobre el subconjunto observable, no sobre el universo completo de ATL

---

## 5. Cómo mencionar esto en la tesis

**Formulación sugerida:**

> El sistema genera predicciones para aproximadamente 939 vuelos únicos diarios en ATL, representando el ~35% de las operaciones totales del aeropuerto. Esta cobertura parcial se debe a las limitaciones de paginación de la AeroAPI de FlightAware con el plan de API utilizado (MAX_PAGES=7, ~22 llamadas por ciclo de 30 minutos). Como consecuencia, las features de linaje de cola del avión presentan una tasa de valores faltantes del 71.6%, dado que el vuelo anterior del mismo avión frecuentemente no fue capturado. El modelo LightGBM maneja estos valores faltantes de forma nativa. El AUC reportado (0.82) corresponde al subconjunto de vuelos observados por el sistema, no al universo completo de operaciones de ATL. Para cobertura completa, la integración de OpenSky Network como fuente de actuals y queries segmentadas por aerolínea en AeroAPI serían las mejoras de mayor impacto con menor costo adicional.
