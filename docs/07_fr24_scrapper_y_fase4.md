# FR24 Scrapper — Cobertura de campos, estado de infraestructura y Fase 4

**Fecha de análisis**: 2026-08-12  
**Contexto**: Análisis de si el scrapper de FlightRadar24 ya cubre todos los campos que necesita el modelo LightGBM v9 y qué falta para activarlo como fuente principal.

---

## 1. Infraestructura de recolección actual

### Jobs y schedulers en producción (GCP proyecto `ontimeai`)

| Job (Cloud Run Job) | Scheduler | Frecuencia | Función |
|---|---|---|---|
| `ontimeai-live-pull` | `ontimeai-pull-scheduler` | `*/30 * * * *` | Pipeline de predicción principal (AeroAPI modo) |
| `ontimeai-live-pull-2` | (segundo scheduler) | `*/30 * * * *` (offset +15 min) | Pipeline secundario, misma imagen que `live-pull` |
| `ontimeai-harvester` | (scheduler propio) | `*/30 * * * *` (`:08/:38`) | Scrapper FR24 — ingesta de vuelos y chain walk |
| `ontimeai-outcome-backfill` | (scheduler propio) | periódico | Backfill de actuals desde AeroAPI |

En total: **4 Cloud Run Jobs**, **3+ schedulers**, corriendo todos en paralelo.

### Repositorio del scrapper

```
/Users/lologalaverna/Projects/Tesis/OnTimeAI-Scrapper/
├── ontimeai_scrapper/
│   ├── fr24_client.py          ← cliente FR24 con curl_cffi + chrome136
│   ├── chain_walk.py           ← Capa 2: hidratación de inbound_fa_flight_id
│   └── harvester_job.py        ← entrypoint del Cloud Run Job
```

**Librería**: `FlightRadarAPI v1.5.1` con `curl_cffi` + fingerprint `chrome136` para bypassear Cloudflare. **Costo estimado**: ~$0.10/día (solo compute de Cloud Run).

---

## 2. Por qué el lineaje sigue vacío en el pipeline de predicción

### El problema

El pipeline de predicción (`live_pull.py`) corre en modo `LIVE_DATA_SOURCE=aeroapi` (default). AeroAPI tiene **~35% de cobertura de vuelos ATL** del total de tráfico real, con un 71.6% de `inbound_fa_flight_id` nulos (lineaje faltante). Esto degradaba la feature `TAIL_DELAY_DECAY` del modelo.

El scrapper FR24 tiene **92.3% de cobertura de lineaje** pero su data NO es usada por el pipeline de predicción. Ambos corren en paralelo y escriben en la misma DB SQLite (tablas distintas), pero `live_pull.py` ignora la tabla `fr24_flights` del harvester.

### Causa raíz

**Fase 4 nunca fue implementada**: el código de `live_pull.py` tiene la variable de entorno `LIVE_DATA_SOURCE` con soporte para el modo `harvester`, pero el switch completo (leer de tabla FR24 en vez de llamar AeroAPI para departures) fue diseñado pero nunca activado en producción.

---

## 3. Cobertura de campos FR24 vs lo que necesita el modelo v9

### Campos que devuelve `normalize_flight()` en `fr24_client.py`

```python
{
    "fa_flight_id":         str,          # ID interno FR24 (hex 8 chars, ej: "2a5f8b3c")
    "ident_iata":           str,          # "DAL1234"
    "op_carrier":           str,          # "DL"
    "flight_number":        str,          # "1234"
    "tail_num":             str,          # "N12345"
    "origin":               str,          # "ATL"
    "dest":                 str,          # "JFK"
    "inbound_fa_flight_id": None,         # hidratado después por chain walk (Capa 2)
    "scheduled_out_utc":    str,          # ISO8601 UTC
    "scheduled_in_utc":     str,          # ISO8601 UTC
    "estimated_out_utc":    str | None,
    "estimated_in_utc":     str | None,
    "aircraft_type":        str,          # "B737"
    "cancelled":            bool,
    "diverted":             bool,
    "crs_elapsed_min":      int | None,
    "distance":             None,         # FR24 no lo expone consistentemente
}
```

### Comparación con AeroAPI y con los features del modelo

| Campo | AeroAPI | FR24 (scrapper) | Modelo v9 lo usa | Observación |
|---|---|---|---|---|
| `origin`, `dest` | ✅ | ✅ | ✅ | Base para BEARING_DEG, PageRank |
| `scheduled_out_utc` | ✅ | ✅ | ✅ | `crs_dep_min`, hora del día |
| `scheduled_in_utc` | ✅ | ✅ | ✅ | `crs_elapsed_min` derivado |
| `estimated_out_utc` | ✅ | ✅ | ✅ | delay features pre-despegue |
| `estimated_in_utc` | ✅ | ✅ | ✅ | delay features en vuelo |
| `aircraft_type` | ✅ | ✅ | ✅ | `AIRCRAFT_FAMILY` feature |
| `tail_num` | ✅ | ✅ | ✅ | chain walk de lineaje |
| `op_carrier` | ✅ | ✅ | ✅ | `carrier_delay_rate_24h` |
| `cancelled` | ✅ | ✅ | ✅ | filtro pre-inferencia |
| `inbound_fa_flight_id` | ✅ directo | ✅ vía chain walk | ✅ | FR24 = 92.3% vs AeroAPI = 28.4% |
| `distance` | ✅ | ❌ `None` | ✅ | **Ver nota abajo** |
| `crs_elapsed_min` | ✅ | ✅ | ✅ | tiempo de vuelo programado |

#### Nota sobre `distance`

FR24 devuelve `distance=None`. El modelo usa distancia como feature (`DISTANCE`). En el pipeline actual (`live.py`), si `distance` es nulo, se computa desde una tabla de coordenadas de aeropuertos (lookup por IATA code con fórmula haversine). Esta lógica ya existe en el código — la columna nula de FR24 no impacta la calidad del modelo.

### Conclusión de cobertura

**FR24 cubre todos los campos que el modelo necesita.** La única diferencia con AeroAPI es el mecanismo de lineaje: AeroAPI lo provee directo, FR24 lo construye por chain walk (hit rate superior: 92.3% vs 28.4%).

---

## 4. Incompatibilidad de IDs

**Este es el único gap técnico relevante.**

- AeroAPI `fa_flight_id`: formato `IDENT-TIMESTAMP-TYPE-HASH` (ej: `DAL1458-1786401347-sw-970p`)
- FR24 `fa_flight_id`: hex 8 chars (ej: `2a5f8b3c`)

Los IDs son **incompatibles**. No hay join posible entre vuelos AeroAPI y vuelos FR24 via `fa_flight_id`. Si se activa la Fase 4, los vuelos nuevos tendrán IDs FR24 y los históricos tendrán IDs AeroAPI — la tabla `predictions` continuará acumulando ambos sin conflicto, pero los links externos a FlightAware (que usan el ID AeroAPI) deberán reconstruirse desde `ident_iata + fecha`.

---

## 5. Ventajas de activar FR24 como fuente principal (Fase 4)

| Métrica | Estado actual (AeroAPI) | Con Fase 4 (FR24) |
|---|---|---|
| Cobertura de vuelos ATL | ~35% | ~92% |
| Lineaje `inbound_fa_flight_id` | 28.4% | 92.3% |
| Rate limit riesgo | Alto (22 calls/ciclo, saturaba en MAX_PAGES=10) | Ninguno (scraping libre) |
| Costo por llamada | $0.005/call AeroAPI | $0 (scraping) |
| Dependencia de clave API | Sí (Secret Manager `aeroapi-key`) | No |
| Lead time de predicción | ~1.3h antes de salida | ~3-4h estimado (mayor cobertura) |

---

## 6. Cómo activar la Fase 4 (cuando corresponda)

**Prerrequisito**: confirmar que `live_pull.py` en modo `harvester` lee correctamente de la tabla FR24 y construye el inference frame con los mismos features que en modo AeroAPI. Esto requiere una prueba en staging o una ejecución manual controlada.

**El switch**: solo dos comandos gcloud:

```bash
gcloud run jobs update ontimeai-live-pull \
  --update-env-vars="LIVE_DATA_SOURCE=harvester" \
  --region=us-central1 --project=ontimeai

gcloud run jobs update ontimeai-live-pull-2 \
  --update-env-vars="LIVE_DATA_SOURCE=harvester" \
  --region=us-central1 --project=ontimeai
```

**El rollback**: revertir `LIVE_DATA_SOURCE` a `aeroapi` (o eliminarlo, ya que `aeroapi` es el default).

**No requiere redespliegue** — solo cambio de variable de entorno. El job toma el nuevo valor en la próxima ejecución del scheduler.

---

## 7. Estado al 2026-08-14 (actualizado)

- ✅ Scrapper FR24 corriendo en producción, 0 failures en 60+ horas, costo ~$0.10/día
- ✅ Datos FR24 almacenados en `live_data.db` (misma tabla `flights`, mismo schema que backend)
- ✅ Chain walk de lineaje operativo (92.3% hit rate, 2400+ tails en cache)
- ✅ **Fase 4 activada en producción el 2026-08-13** — `LIVE_DATA_SOURCE=harvester` en ambos jobs
- ✅ Pipeline de predicción en modo harvester: 0 AeroAPI calls por ciclo, costo $0
- ✅ `TAIL_DELAY_DECAY` NaN rate bajó de 71.6% → 1.6% (lineaje casi completo)
- ✅ Frecuencia live-pull: cada 30 min (scheduler estándar), harvester cada 15 min
- ✅ `FR24_MAX_PAGES=30` — valor definitivo (ver nota abajo)

### Nota sobre MAX_PAGES

El 2026-08-14 se probó `FR24_MAX_PAGES=60`. Resultado: el harvester siguió llenando todas las páginas (1–60), pero el conteo de vuelos ATL no aumentó significativamente (193–203 vuelos vs 178–208 con 30 páginas). La ganancia fue de ~5-10 vuelos extra al precio de duplicar la duración del ciclo (104s vs 51s).

**Conclusión**: el cuello de botella no es la paginación sino el volumen real que FR24 expone para ATL en una ventana de 6h. El techo de cobertura de FR24 es ~92% del tráfico comercial de ATL. El 8% restante corresponde a aviación general, charters sin código IATA, vuelos militares y aeronaves sin transponder visible — no recuperables desde ninguna fuente pública de scraping.

`FR24_MAX_PAGES` fue revertido a 30 el mismo día.
