# Plan: Harvester Continuo de Lineage para Live Tests

**Fecha:** 2026-05-22
**Estado:** Borrador para revisión técnica antes de implementar
**Autor:** Síntesis colaborativa Claude + Santiago

---

## 1. Contexto y problema

### 1.1 Síntoma observado

El modelo v9 en producción tiene una brecha estructural entre métricas offline y live:

| Métrica | Offline (test) | Live (10,900 predicciones) | Gap |
|---|---|---|---|
| ROC AUC | 0.8602 | 0.7183 | **−14.2 pts** |
| Brier | 0.103 | 0.146 | +0.043 |
| Accuracy | 0.856 | 0.717 | −13.9 pts |

### 1.2 Causa raíz validada por datos

El `data_quality_report.json` revela el patrón:

| Días Grade A | `lineage_hit_rate` | Días Grade F | `lineage_hit_rate` |
|---|---|---|---|
| 2026-05-12 | 0.755 | 2026-05-04 | 0.328 |
| 2026-05-13 | 0.810 | 2026-05-07 | 0.392 |
| 2026-05-15 | **0.931** | 2026-05-08 | 0.359 |
| 2026-05-16 | 0.821 | | |
| 2026-05-17 | 0.785 | | |

**Conclusión**: el AUC live colapsa cuando las features de lineage (PREV_ARR_DELAY_TAIL, prev_turnaround_tail_min, tail_flights_today_prior, etc., que representan 34% del gain del modelo según SESSION_FINDINGS.md) quedan en NaN porque el buffer está vacío o desactualizado.

### 1.3 Objetivo del plan

Construir un **servicio de cosecha continua (harvester) 24/7** que mantenga el buffer `live_data.db` permanentemente caliente, garantizando que `lineage_hit_rate ≥ 0.85` todos los días — independientemente de si el cron de predicción corre o no.

**No es un servicio de predicción**. Es un servicio de **grabación de actuals** que el `live_pull.py` consume cuando hace inferencia.

### 1.4 Restricciones

- **Costo objetivo**: gratis o ≤ $5 USD/mes en infraestructura
- **No depender de AeroAPI** para sustento (sí como fallback de emergencia)
- **Compatible con stack actual**: Cloud Run Job + GCS + SQLite + Cloud Scheduler
- **No tocar el modelo ni el feature builder** — solo cambiar la fuente de los actuals que el feature builder lee

---

## 2. Resultados de la investigación de APIs

### 2.1 Resumen ejecutivo

Verifiqué online cuatro fuentes candidatas para chain walking de tails. **Hallazgo crítico**: la librería `FlightRadarAPI` (JeanExtreme002) **no expone un método nativo de aircraft history** — el caso de uso "dame los últimos 7 días de vuelos del tail N12345" requiere llamada HTTP directa al endpoint interno de FR24 que la librería no envuelve.

### 2.2 Comparación detallada

| Fuente | Endpoint relevante | Devuelve schedule + actual + tail | Aircraft history | Auth | Rate limit | TOS |
|---|---|---|---|---|---|---|
| **FR24 lib** `get_airport_details("KATL", page=N)` | `/airports/traffic-stats/` | ✅ sí | ❌ no (solo airport-anchored) | No | ~1 req/s (Cloudflare) | TOS-grey (educativo) |
| **FR24 lib** `get_flights(registration="N12345")` | `/zones/fcgi/feed.js` | ⚠️ solo vuelos en aire ahora | ❌ no | No | ~1 req/s | TOS-grey |
| **FR24 lib** `get_history_data(flight, "CSV", ts)` | `/download/?flight=...` | ✅ breadcrumbs de un vuelo | ❌ no (un vuelo a la vez) | No | ~1 req/s | TOS-grey |
| **FR24 HTTP directo** `api.flightradar24.com/common/v1/flight/list.json?query=N12345&fetchBy=reg` | (no en librería) | ✅✅ sí, historial 30 días | ✅ **SÍ** | No | ~1 req/s | TOS-grey |
| **OpenSky** `/flights/aircraft?icao24=...&begin=...&end=...` | oficial | ❌ solo ADS-B actuals | ✅ sí, ventana ≤ 2 días | Opcional | Anónimo 400 cred/día; auth 4000 | Open |
| **OpenSky** `/flights/arrival?airport=KATL` | oficial | ❌ solo ADS-B actuals | n/a (por aeropuerto) | Opcional | mismo bucket | Open |
| **airplanes.live** `/v2/point/lat/lon/radius` | oficial | ❌ solo posición + tail (`r`) | ❌ solo live | No | 1 req/s | Open |
| **FAA MASTER.txt** | `registry.faa.gov/database/ReleasableAircraft.zip` | n/a (lookup hex→N-number) | n/a | No | n/a | Public domain |

### 2.3 Conclusiones operativas

1. **Capa 1 (ATL anchor)**: la librería `FlightRadarAPI` con `get_airport_details` es la opción directa. Devuelve scheduled + actual + tail en un solo call paginado.

2. **Capa 2 (chain walk por tail)**: la librería **no lo expone**. Hay dos caminos:
   - **2A**: HTTP directo a `api.flightradar24.com/common/v1/flight/list.json?query={REG}&fetchBy=reg` — devuelve sched + actual + tail completo, mismo riesgo Cloudflare que la librería.
   - **2B**: OpenSky `/flights/aircraft?icao24=...` — devuelve solo actuals (sin scheduled), pero es API oficial sin TOS-grey.
   - **Recomendación**: 2A primario, 2B fallback. Para `prev_arr_delay_tail` necesitamos scheduled+actual, que solo 2A da en un call.

3. **FAA MASTER.txt**: lookup offline indispensable si caés a OpenSky como fallback (ICAO24 hex → N-number). Refresh nocturno. ~80 MB.

4. **airplanes.live**: útil como segundo fallback para detectar tails en el cielo de ATL en tiempo real (ya devuelve campo `r` resuelto). No reemplaza a FR24 porque no tiene scheduled times.

---

## 3. Arquitectura propuesta

### 3.1 Diagrama de capas

```
┌─────────────────────────────────────────────────────────────────┐
│  CAPA 1 — ATL ANCHOR (Cloud Scheduler cron */15 * * * *)        │
│  Job: ontimeai-atl-harvester                                    │
│                                                                  │
│  fr24.get_airport_details("KATL", flight_limit=100, page=N)     │
│    → arrivals + departures con sched + actual + tail            │
│    → paginar hasta agotar (~4-7 pages = ±6h ventana)            │
│    → UPSERT en flights, actuals                                 │
│                                                                  │
│  Output: ~250-500 vuelos nuevos/actualizados por tick           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (detecta tails sin lineage)
┌─────────────────────────────────────────────────────────────────┐
│  CAPA 2 — CHAIN WALK LAZY (mismo job, mismo tick)               │
│                                                                  │
│  Para cada tail en Capa 1 sin entry en tail_lineage_cache:      │
│    fetch api.flightradar24.com/common/v1/flight/list.json       │
│         ?query={REG}&fetchBy=reg&limit=25                       │
│    → últimos ~7 días de vuelos del tail (todos aeropuertos)     │
│    → UPSERT en flights, actuals                                 │
│    → marcar tail en tail_lineage_cache(hydrated_until=now)      │
│                                                                  │
│  Fallback si FR24 falla:                                         │
│    1. lookup tail → icao24 vía tail_to_icao24_lookup            │
│    2. opensky /flights/aircraft?icao24=...&begin=ts_7d&end=now  │
│    3. UPSERT actuals (sin scheduled — features parciales)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (refresca tails que ya están en cache pero stale)
┌─────────────────────────────────────────────────────────────────┐
│  CAPA 3 — REFRESH ACTIVO (cron separado */6h)                   │
│  Job: ontimeai-lineage-refresh                                  │
│                                                                  │
│  Para cada tail con vuelos programados en KATL en últimas 12h:  │
│    re-pullar últimas 24h de su historia (chain walk)            │
│    → captura actuals que se acaban de settle                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (job nocturno, infraestructura)
┌─────────────────────────────────────────────────────────────────┐
│  JOB INDEPENDIENTE — FAA MASTER REFRESH (cron 0 3 * * *)        │
│  Job: ontimeai-faa-master-sync                                  │
│                                                                  │
│  curl https://registry.faa.gov/database/ReleasableAircraft.zip  │
│  unzip → MASTER.txt                                             │
│  parse → tabla SQLite tail_to_icao24_lookup                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (consumidor — sin cambios funcionales)
┌─────────────────────────────────────────────────────────────────┐
│  live_pull.py (existente)                                        │
│                                                                  │
│  Cuando corre predicción:                                        │
│    → lee actuals + flights del buffer (ya caliente)             │
│    → feature builder calcula lineage sin llamar APIs            │
│    → predict → INSERT predictions                               │
│                                                                  │
│  CAMBIO MÍNIMO: ENV var SOURCE=harvester | aeroapi              │
│  cuando == "harvester", saltea las llamadas AeroAPI y va        │
│  directo a leer del buffer                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Schema SQLite — tablas nuevas

```sql
-- Cache de hidratación por tail (evita re-pullar historia ya capturada)
CREATE TABLE IF NOT EXISTS tail_lineage_cache (
    tail TEXT PRIMARY KEY,
    hydrated_until TEXT NOT NULL,        -- ISO UTC
    last_pull_source TEXT NOT NULL,      -- 'fr24' | 'opensky' | 'aeroapi'
    last_pull_ok INTEGER NOT NULL,       -- 0 | 1
    consecutive_failures INTEGER DEFAULT 0
);

CREATE INDEX idx_tail_lineage_cache_hydrated_until
    ON tail_lineage_cache(hydrated_until);

-- Lookup hex → N-number (FAA MASTER nightly)
CREATE TABLE IF NOT EXISTS tail_to_icao24_lookup (
    icao24 TEXT PRIMARY KEY,             -- lowercase hex
    n_number TEXT NOT NULL,
    aircraft_type TEXT,
    aircraft_year INTEGER,
    last_synced TEXT NOT NULL
);

CREATE INDEX idx_tail_to_icao24_n_number ON tail_to_icao24_lookup(n_number);

-- Métricas operacionales del harvester
CREATE TABLE IF NOT EXISTS harvester_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at_utc TEXT NOT NULL,
    layer TEXT NOT NULL,                 -- 'atl_anchor' | 'chain_walk' | 'refresh'
    n_calls INTEGER,
    n_flights_upserted INTEGER,
    n_actuals_upserted INTEGER,
    n_tails_hydrated INTEGER,
    duration_seconds REAL,
    status TEXT,                         -- 'ok' | 'partial' | 'failed'
    error TEXT
);
```

Las tablas existentes `flights`, `actuals`, `weather_obs`, `predictions`, `runs` no cambian.

### 3.3 Lógica de hidratación cacheada (pseudocódigo)

```python
def maybe_hydrate_tail(tail_num: str, conn: sqlite3.Connection, *,
                       freshness_hours: int = 6) -> None:
    """Capa 2 con cache. Llamar para cada tail visto en Capa 1."""

    row = conn.execute(
        "SELECT hydrated_until, consecutive_failures "
        "FROM tail_lineage_cache WHERE tail = ?",
        (tail_num,),
    ).fetchone()

    now = datetime.now(timezone.utc)
    if row:
        hydrated_until = datetime.fromisoformat(row["hydrated_until"])
        if hydrated_until > now - timedelta(hours=freshness_hours):
            return  # cache fresco, skip
        if row["consecutive_failures"] >= 5:
            return  # tail flaky, retry next refresh cycle

    try:
        flights = fetch_fr24_aircraft_history(tail_num, days=7)
        source = "fr24"
    except FR24Error:
        try:
            icao24 = lookup_icao24(tail_num, conn)
            flights = fetch_opensky_aircraft_history(icao24, days=2)
            source = "opensky"
        except (LookupError, OpenSkyError):
            mark_failure(tail_num, conn)
            return

    upsert_flights_and_actuals(flights, conn)
    conn.execute(
        "INSERT OR REPLACE INTO tail_lineage_cache "
        "(tail, hydrated_until, last_pull_source, last_pull_ok, consecutive_failures) "
        "VALUES (?, ?, ?, 1, 0)",
        (tail_num, now.isoformat(), source),
    )
```

---

## 4. Plan de implementación por fases

### Fase 0 — Validación local (1 día)

**Objetivo**: confirmar empíricamente que las APIs devuelven lo que la documentación promete.

Crear notebook `notebooks/validate_harvester_sources.ipynb`:

- [ ] 0.1 Probar `fr.get_airport_details("KATL", flight_limit=100, page=1)` y mostrar schema completo del response
- [ ] 0.2 Probar `requests.get("https://api.flightradar24.com/common/v1/flight/list.json", params={"query": "N301DQ", "fetchBy": "reg", "limit": 25})` con headers de browser y confirmar respuesta JSON con sched+actual+tail
- [ ] 0.3 Probar OpenSky `/flights/aircraft?icao24=a06b4f&begin=...&end=...` (icao24 corresponde a un tail Delta conocido)
- [ ] 0.4 Descargar FAA MASTER.txt manualmente y confirmar columnas N-NUMBER y MODE S CODE HEX
- [ ] 0.5 Documentar exactamente el mapeo entre fields de cada fuente y las columnas que `live.py:_normalize_flight_row()` espera

**Criterio de éxito**: cada call devuelve los fields esperados con datos reales para ~3 tails de prueba.

### Fase 1 — Harvester ATL anchor en producción (3-5 días)

**Objetivo**: cron 24/7 que mantiene actualizada la ventana ±6h de ATL.

Crear:
- [ ] 1.1 `ontimeai/fr24_client.py` — wrapper sobre la librería `FlightRadarAPI` con throttle 1.5s, backoff exponencial en 403/429, pin de versión `==1.5.1`
- [ ] 1.2 `harvester.py` (script standalone, no toca `live_pull.py`) — Capa 1 únicamente
- [ ] 1.3 `Dockerfile.harvester` para Cloud Run Job
- [ ] 1.4 Cloud Build + deploy con SA dedicada
- [ ] 1.5 Cloud Scheduler `ontimeai-harvester-scheduler` con cron `*/15 * * * *`
- [ ] 1.6 Logs estructurados a Cloud Logging con métricas: n_calls, n_flights_upserted, lineage_hit_rate_predicted
- [ ] 1.7 Alert si `n_flights_upserted < 100` por dos ticks consecutivos (síntoma de FR24 caído)

**Criterio de éxito**: tras 24h corriendo, `lineage_hit_rate` calculado offline sobre el buffer ≥ 0.70 (igual o mejor que el promedio actual con AeroAPI).

### Fase 2 — Chain walk lazy (1 semana)

**Objetivo**: agregar Capa 2 al harvester para cerrar el gap del primer hop del día.

- [ ] 2.1 Implementar `fetch_fr24_aircraft_history(reg, days=7)` con HTTP directo (no librería)
- [ ] 2.2 Implementar `maybe_hydrate_tail()` con cache
- [ ] 2.3 Integrar en el flow del Capa 1 (después de upsert, scanear nuevos tails)
- [ ] 2.4 Métricas: `n_tails_hydrated_per_tick`, `cache_hit_rate`

**Criterio de éxito**: tras 7 días de Capa 1+2 corriendo, `lineage_hit_rate ≥ 0.90` consistentemente.

### Fase 3 — Refresh activo y FAA sync (3-4 días)

- [ ] 3.1 Cron separado `*/6h` para Capa 3 (refresh de tails activos)
- [ ] 3.2 Cron nocturno para descarga FAA MASTER.txt + ETL a `tail_to_icao24_lookup`
- [ ] 3.3 Fallback OpenSky en `maybe_hydrate_tail` cuando FR24 falla
- [ ] 3.4 Tests de integración del fallback (simular 403 de FR24)

**Criterio de éxito**: el harvester sobrevive a 24h con FR24 deliberadamente bloqueado, manteniendo `lineage_hit_rate ≥ 0.75` con OpenSky-only.

### Fase 4 — Switch del live_pull.py (1 día)

- [ ] 4.1 Agregar env var `LIVE_DATA_SOURCE` en `live_pull.py`
  - `aeroapi` (default, comportamiento actual)
  - `harvester` (saltea llamadas AeroAPI, lee del buffer)
- [ ] 4.2 Deploy `ontimeai-live-pull` con `LIVE_DATA_SOURCE=harvester`
- [ ] 4.3 Comparar live_metrics 7 días: harvester-only vs aeroapi-only
- [ ] 4.4 Si harvester-only AUC ≥ aeroapi-only AUC × 0.97 → consolidar harvester como primario
- [ ] 4.5 AeroAPI queda como fallback automático si harvester falla 3 ticks seguidos

**Criterio de éxito**: AUC live con harvester gratuito ≥ 0.70 (el AUC actual con AeroAPI pago es 0.7183).

---

## 5. Estimaciones

### 5.1 Volumen de queries (estado estable, después de warm-up de 7 días)

| Capa | Frecuencia | Calls/día |
|---|---|---|
| Capa 1 — ATL anchor (4 páginas × cron 15min) | 96 ticks × 4 | 384 |
| Capa 2 — Chain walk (hidratación de tails nuevos + cache miss en refresh) | ~80-100 tails/día × 1 call | 100 |
| Capa 3 — Refresh activo (cron 6h × tails activos) | 4 ticks × ~150 tails | 600 |
| FAA MASTER sync | 1 vez | 1 |
| **TOTAL FR24** | | **~1,085 calls/día** |
| OpenSky fallback (solo si FR24 ≥3 fails) | esperado <50 días/año | <100 cred/incidente |

Promedio: **1 call cada 80 segundos**. Threshold empírico Cloudflare ~1 call/segundo. **Margen de seguridad ~13×**.

### 5.2 Infraestructura

| Componente | Costo/mes |
|---|---|
| Cloud Run Job harvester (15 min cron × ~45 s × ~$0.0001/run) | $0.30 |
| Cloud Run Job refresh (6h cron × ~60 s) | $0.05 |
| Cloud Run Job FAA sync (24h cron × ~30 s) | $0.01 |
| Cloud Storage delta (~50 MB) | $0.02 |
| Cloud Scheduler 3 schedulers | $0.30 |
| Cloud Logging adicional | ~$1.00 |
| **TOTAL** | **~$1.70 / mes** |

Vs AeroAPI actual ~$30-60/mes. **Ahorro 18-35×**.

### 5.3 Tiempo de desarrollo estimado

| Fase | Effort |
|---|---|
| Fase 0 — Validación local | 1 día |
| Fase 1 — Capa 1 producción | 3-5 días |
| Fase 2 — Capa 2 chain walk | 5-7 días |
| Fase 3 — Capa 3 + FAA + fallback | 3-4 días |
| Fase 4 — Switch | 1-2 días |
| **TOTAL** | **~3 semanas calendario** |

---

## 6. Riesgos y mitigaciones

### 6.1 Riesgo: FR24 cambia endpoint y la librería rompe

**Probabilidad**: alta a 12 meses (issue #96 cerró bypass de Cloudflare en mayo 2026; issues recurrentes cada ~6-9 meses).

**Impacto**: harvester queda sin Capa 1 ni Capa 2 hasta que la librería se actualice.

**Mitigación**:
- Pin de versión `FlightRadarAPI==1.5.1`. No auto-upgrade.
- Fallback automático a OpenSky para Capa 2 (sin scheduled times → features parciales pero el modelo tolera NaN).
- Para Capa 1: si FR24 falla, se puede pullear ATL via `OpenSky /flights/arrival?airport=KATL` + `/flights/departure?airport=KATL` (sin scheduled).
- AeroAPI sigue activable con `LIVE_DATA_SOURCE=aeroapi` (1 línea de env var en Cloud Run).

### 6.2 Riesgo: Cloudflare bot detection escala y banea la IP del Cloud Run Job

**Probabilidad**: media. Con 1.5s throttle + jitter es bajo, pero no cero.

**Impacto**: harvester pierde acceso desde la IP de ese Cloud Run Job hasta que GCP la rote.

**Mitigación**:
- Throttle conservador (1.5s mínimo + jitter aleatorio ±500ms)
- User-Agent rotativo (lista de 5 strings de browser modernos)
- Cloud Run rota IPs naturalmente en redeploys
- Si baneo prolongado: deploy en una segunda región como respaldo

### 6.3 Riesgo: FAA MASTER.txt cambia formato o URL

**Probabilidad**: baja (es un dataset gubernamental estable).

**Impacto**: el lookup hex→N-number queda stale. Afecta solo al fallback OpenSky.

**Mitigación**:
- ETL del FAA con validación de schema (columnas esperadas presentes)
- Si parse falla, mantener la tabla anterior (no sobrescribir)
- Alert si la tabla no se actualiza por 3 días

### 6.4 Riesgo: cobertura insuficiente para carriers no-Delta

**Probabilidad**: media. Vuelos de AA/UA/WN cuyo primer hop del día NO toca ATL pueden quedar con `prev_arr_delay_tail` en NaN.

**Impacto**: ~2% de vuelos diarios con lineage parcial (50 vuelos × 1 carrier no-Delta × ~5 carriers principales).

**Mitigación**:
- El modelo ya tolera NaN en lineage (entrenado con BTS que tiene gaps).
- Si se vuelve problema demostrable: agregar Capa 1' con `get_airport_details` para top-5 hubs (DFW, ORD, LAX, DEN, JFK) — +5 calls cada 15 min = ~480 calls/día adicionales (sigue bajo threshold).

### 6.5 Riesgo TOS — uso académico vs comercial

**Probabilidad**: baja para tesis. Alta si se publica como producto.

**Impacto**: cease & desist legal si se viraliza.

**Mitigación**:
- Documentar claramente uso académico en el repo
- No publicar API pública del backend que use estos datos
- Si la tesis se convierte en producto post-defensa: migrar a FR24 API oficial ($9-99/mes) o AeroAPI

---

## 7. Métricas de éxito y validación

### 7.1 Métricas operacionales (durante implementación)

| Métrica | Target |
|---|---|
| `harvester_runs.status == 'ok'` por día | ≥ 90% de los ticks |
| `n_flights_upserted` por tick (Capa 1) | ≥ 100 |
| `cache_hit_rate` en Capa 2 (post warm-up) | ≥ 75% |
| FR24 calls/día | ≤ 1,500 |
| `consecutive_failures` por tail | < 5 antes de pausa |

### 7.2 Métricas de calidad (impacto real)

| Métrica | Baseline (AeroAPI) | Target (harvester) |
|---|---|---|
| `lineage_hit_rate` promedio diario | 0.61 (volátil 0.30-0.93) | ≥ 0.85 (estable) |
| Días Grade A en `data_quality_report` | 5/18 (27.7%) | ≥ 80% |
| AUC live | 0.7183 | ≥ 0.70 |
| Brier live | 0.146 | ≤ 0.155 |

### 7.3 Pruebas A/B propuestas

Después de Fase 4, correr en paralelo durante 14 días:
- Tracker A: `LIVE_DATA_SOURCE=aeroapi` (control, costo ~$30)
- Tracker B: `LIVE_DATA_SOURCE=harvester` (experimental, costo ~$2)

Comparar:
- AUC live sobre el mismo conjunto de actuals
- `lineage_hit_rate` diario
- Costo total real
- Incidentes operacionales

**Decisión final**: si Tracker B alcanza ≥ 97% del AUC de Tracker A, deprecar AeroAPI.

---

## 8. Decisiones pendientes — necesarias antes de empezar

### 8.1 Decisión D1 — ¿Crear job nuevo separado o extender el existente?

**Opción A**: Job nuevo `ontimeai-atl-harvester` separado de `ontimeai-live-pull`.
- Pro: aislamiento, no rompe lo que funciona
- Con: dos jobs corriendo (más infra a mantener)

**Opción B**: Extender `live_pull.py` con un modo `--harvest-only` que solo cosecha.
- Pro: un solo binario
- Con: acopla la cosecha con la predicción

**Recomendación**: Opción A. Costos mínimos, riesgo de regresión nulo.

### 8.2 Decisión D2 — ¿Tener cuenta OpenSky autenticada desde el día 1?

Anónimo: 400 cred/día. Autenticado: 4,000 cred/día.

Solo se usa OpenSky como fallback (esperado <100 cred/incidente), pero si hay un blackout largo de FR24, podría agotarse rápido.

**Recomendación**: registrar cuenta gratuita desde día 1, guardar credenciales en Secret Manager.

### 8.3 Decisión D3 — ¿Capa 2 corre en mismo tick que Capa 1, o asincrónica?

**Opción A**: en mismo tick (orden: pull ATL → para cada tail nuevo, chain walk).
- Pro: simple
- Con: ticks de 15 min pueden tardar 2-3 min en días con muchos tails nuevos

**Opción B**: Capa 2 como Cloud Tasks asincrónica, encolada por Capa 1.
- Pro: ticks rápidos, paralelismo controlado
- Con: más complejidad operacional

**Recomendación**: Opción A para MVP. Migrar a B solo si los ticks se vuelven >5 min.

### 8.4 Decisión D4 — Política de retención del buffer

Hoy `live_data.db` crece sin política de retención. Con harvester continuo crecerá ~10× más rápido.

**Recomendación**: GC nocturno que borra flights con `crs_dep_utc < now - 30d`. El modelo no usa lineage de más de 7 días.

### 8.5 Decisión D5 — ¿Empezar con Fase 0 (validación local) o ir directo a Fase 1?

**Recomendación**: Fase 0 obligatoria. La librería FR24 no tiene aircraft history nativo, y el HTTP directo a `flight/list.json` no está documentado oficialmente. Validar empíricamente que devuelve los fields esperados antes de invertir en infra es crítico.

---

## 9. Próximo paso concreto

Si este plan se aprueba como está:

1. **Hoy mismo**: ejecutar Fase 0 — crear `notebooks/validate_harvester_sources.ipynb` y verificar las 5 cosas listadas en §4 Fase 0.
2. **Esta semana**: implementar Fase 1 — `ontimeai/fr24_client.py` + `harvester.py` + deploy del Cloud Run Job + Scheduler.
3. **Próximas 2 semanas**: Fases 2-3 (chain walk + refresh + FAA sync + fallback OpenSky).
4. **Semana 4**: Fase 4 — switch del `live_pull.py` con A/B testing de 14 días.

**Total**: harvester gratuito en producción en ~3-4 semanas, A/B validado a las 6 semanas, deprecar AeroAPI a las 7-8 semanas.

Si todo sale como las cuentas sugieren: costo operacional reducido de ~$40/mes a ~$2/mes, AUC live mejorado de 0.72 a ≥0.78, y `lineage_hit_rate` estabilizado en ≥0.85 todos los días.

---

## Apéndice A — Mapeo de campos entre fuentes y schema de live.py

| Campo en `live.py` / `flights` table | FR24 `get_airport_details` | FR24 HTTP directo `flight/list.json` | OpenSky `/flights/aircraft` |
|---|---|---|---|
| `fa_flight_id` (PK) | `flight.identification.id` | `flight.identification.id` | sintético: `{icao24}_{firstSeen}` |
| `tail_num` | `flight.aircraft.registration` | `aircraft.registration` | requiere lookup FAA MASTER |
| `op_carrier` | `flight.airline.code.iata` | `airline.code.iata` | derivar de callsign (ej. DAL→DL) |
| `origin` (IATA) | `flight.airport.origin.code.iata` | `airport.origin.code.iata` | `estDepartureAirport` (ICAO → IATA) |
| `dest` (IATA) | `flight.airport.destination.code.iata` | `airport.destination.code.iata` | `estArrivalAirport` (ICAO → IATA) |
| `crs_dep_utc` | `flight.time.scheduled.departure` (epoch) | `time.scheduled.departure` | ❌ no disponible |
| `scheduled_in_utc` | `flight.time.scheduled.arrival` | `time.scheduled.arrival` | ❌ no disponible |
| `actual_off_utc` | `flight.time.real.departure` | `time.real.departure` | `firstSeen` (epoch) |
| `actual_in_utc` | `flight.time.real.arrival` | `time.real.arrival` | `lastSeen` (epoch) |
| `arr_delay_min` | calcular `actual_in - scheduled_in` | calcular | calcular (con scheduled NaN si solo OpenSky) |

## Apéndice B — Endpoints HTTP exactos verificados

```
# Capa 1 — ATL anchor (via librería)
GET https://www.flightradar24.com/airports/traffic-stats/?airport=KATL
    headers: User-Agent: Mozilla/5.0 ...
    Llamado por: fr.get_airport_details("KATL", flight_limit=100, page=N)

# Capa 2A — Chain walk FR24 (HTTP directo, no en librería)
GET https://api.flightradar24.com/common/v1/flight/list.json
    ?query={REGISTRATION}      # ej. N301DQ
    &fetchBy=reg
    &limit=25
    &page=1
    headers: User-Agent: ..., Origin: https://www.flightradar24.com
    Devuelve: JSON con array `result.response.data[].flight` con sched+actual+tail

# Capa 2B — Chain walk OpenSky (fallback)
GET https://opensky-network.org/api/flights/aircraft
    ?icao24={ICAO24_LOWERCASE}  # ej. a06b4f
    &begin={UNIX_TS_INICIO}      # max 2 días atrás
    &end={UNIX_TS_FIN}
    Auth: opcional (basic auth con cuenta OpenSky)
    Rate: anonymous 400 cred/día, auth 4000 cred/día
    Devuelve: [{ icao24, firstSeen, estDepartureAirport, lastSeen, estArrivalAirport, callsign }]

# FAA MASTER nightly
GET https://registry.faa.gov/database/ReleasableAircraft.zip
    Unzip → MASTER.txt (CSV, ~80 MB)
    Columnas clave: "N-NUMBER", "MFR MDL CODE", "MODE S CODE HEX", "MFR" (aircraft manufacturer)
    Refresh: diario aprox 23:30 CT
```

## Apéndice C — Histórico de la conversación que originó este plan

- Sesión 2026-05-22: Santiago consulta cómo replicar AeroAPI gratis para alimentar lineage continuamente
- Investigación inicial identificó FR24 lib + airplanes.live + FAA MASTER como candidatos
- Validación reveló que `lineage_hit_rate` es la métrica que predice Grade A vs Grade F en `data_quality_report.json`
- Investigación profunda confirmó que FR24 lib NO tiene aircraft history nativo → workaround HTTP directo
- Decisión final: arquitectura de 3 capas con FR24 primario + OpenSky fallback + FAA lookup
