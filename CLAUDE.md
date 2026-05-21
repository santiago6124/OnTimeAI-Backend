# OnTimeAI — Contexto de infraestructura para agentes LLM

Este documento describe el proyecto, la infraestructura de producción, los patrones de trabajo y las convenciones para que cualquier LLM pueda operar, depurar y extender el sistema sin necesitar contexto previo de conversaciones anteriores.

---

## 1. Qué es OnTimeAI

**Tesis de grado — UCC Grupo 9.** Sistema de predicción en tiempo real de retrasos de vuelos en el aeropuerto Hartsfield-Jackson Atlanta (ATL/KATL).

- **Modelo**: LightGBM v9 (`4year_v9`), calibrado con datos live (`4year_v9_recal`). Entrenado sobre dataset BTS full-US 2021-2024. ~69 features. AUC test ≈ 0.847, AUC live ≈ 0.718 (gap por distributional shift).
- **Definición de retraso**: `arr_delay_min > 15` minutos (estándar DOT/BTS).
- **Umbral**: estrategia `quantile@0.22` — se recalcula en cada ciclo para mantener ~22% de predicciones positivas.
- **Dato live**: FlightAware AeroAPI v4. Clave en Secret Manager (`aeroapi-key`).

---

## 2. Estructura del repositorio

```
/Users/lologalaverna/Projects/Tesis/
├── deploy.sh                  ← script único de despliegue
├── CLAUDE.md                  ← este archivo
├── OnTimeAI-Backend/          ← FastAPI + pipeline live
│   ├── api.py                 ← servidor HTTP (FastAPI)
│   ├── live_job.py            ← entrypoint del Cloud Run Job
│   ├── live_pull.py           ← pipeline de 1 tick (30 min)
│   ├── ontimeai/live.py       ← lógica de DB, features, merge weather
│   ├── ontimeai/model.py      ← carga de artefactos, predict_proba
│   ├── predict.py             ← prepare_inference_frame()
│   ├── artifacts/4year_v9/    ← modelo activo (booster + calibrador)
│   ├── artifacts/4year_v9_recal/ ← idem, calibrado con live actuals
│   ├── live_data.db           ← SQLite bundleado (fallback arranque)
│   ├── Dockerfile             ← imagen del backend API
│   ├── Dockerfile.job         ← imagen del live job
│   └── .env.local             ← variables locales (no subir a GCS)
└── OnTimeAI-Frontend/         ← Next.js 16 App Router
    ├── src/app/               ← páginas (Server + Client components)
    ├── src/components/        ← componentes reutilizables
    ├── src/lib/api.ts         ← cliente HTTP del backend (server + client side)
    ├── src/lib/auth.ts        ← manejo de JWT (localStorage + cookie)
    ├── Dockerfile             ← imagen del frontend
    └── cloudbuild.yaml        ← Cloud Build config del frontend
```

---

## 3. Infraestructura GCP

**Proyecto GCP**: `ontimeai`
**Región**: `us-central1`
**Service Account predeterminada**: `{PROJECT_NUMBER}-compute@developer.gserviceaccount.com`

### 3.1 Servicios en producción

| Recurso | Nombre GCP | URL / Referencia |
|---|---|---|
| Frontend (Cloud Run) | `ontimeai-frontend` | `https://ontimeai-frontend-hq7henvhjq-uc.a.run.app` |
| Backend (Cloud Run) | `ontimeai-backend` | `https://ontimeai-backend-hq7henvhjq-uc.a.run.app` |
| Live job (Cloud Run Job) | `ontimeai-live-pull` | — |
| Scheduler | `ontimeai-pull-scheduler` | `*/30 * * * *` (cada 30 min) |
| Base de datos | GCS Bucket `ontimeai-live-db` | `gs://ontimeai-live-db/live_data.db` |
| AeroAPI key | Secret Manager | `aeroapi-key:latest` |

### 3.2 Recursos por servicio

| Servicio | CPU | Memoria | Min inst. | Max inst. | Timeout |
|---|---|---|---|---|---|
| Backend | 2 vCPU | 2 GiB | 0 | 2 | 60 s |
| Frontend | 1 vCPU | 512 MiB | 0 | 2 | — |
| Live job | 2 vCPU | 2 GiB | — | — | 600 s |

### 3.3 Base de datos

SQLite en GCS. Flujo:

1. **Live job** descarga `gs://ontimeai-live-db/live_data.db` → `/tmp/live_data.db`
2. Ejecuta `live_pull.py` (pipeline completo)
3. Sube el archivo modificado de vuelta a GCS
4. **Backend** (api.py) refresca su copia local desde GCS cada 30 min (`_DB_REFRESH_INTERVAL = 1800`). En startup descarga inmediatamente.

**Tablas SQLite:**

| Tabla | Contenido |
|---|---|
| `flights` | Vuelos scheduled de AeroAPI. PK: `fa_flight_id` |
| `predictions` | Predicciones LightGBM. PK: `(fa_flight_id, predicted_at_utc)` → acumula historial completo |
| `actuals` | Delays reales de vuelos aterrizados. PK: `fa_flight_id` |
| `weather_obs` | Observaciones METAR de IEM. PK: `(station, valid_utc)` |
| `runs` | Log de ejecuciones del pipeline |

> **IMPORTANTE**: La tabla `predictions` acumula una fila por vuelo por ciclo (30 min). No se sobreescribe ni borra. Es el historial de predicciones.

---

## 4. Pipeline live (live_pull.py)

Se ejecuta cada 30 min. Pasos en orden:

1. **AeroAPI scheduled_departures** — vuelos saliendo de KATL en las próximas **6 horas** (`--schedule-hours 6`)
2. **AeroAPI scheduled_arrivals** — vuelos llegando a KATL en las próximas 6 horas
3. **AeroAPI arrivals/departures completados** — últimas 4 horas para cargar actuals
4. **Chain-walk `inbound_fa_flight_id`** — hidrata lineage (retraso del avión anterior) sin esperar backfill
5. **IEM METAR** — weather de aeropuertos activos del día
6. **Inferencia LightGBM** — construye inference frame, features v9, predice, calibra
7. **INSERT predictions** — guarda en SQLite, sube a GCS

**Variables de entorno del job:**
- `GCS_BUCKET=ontimeai-live-db`
- `ACTIVE_MODEL=4year_v9`
- `AEROAPI_KEY` — desde Secret Manager

---

## 5. Backend API (api.py)

**Base URL**: `https://ontimeai-backend-hq7henvhjq-uc.a.run.app`
**Docs interactivos**: `/docs` (Swagger UI)

### Autenticación

JWT Bearer token. Credenciales por defecto:
- Usuario: `admin` (env `API_USERNAME`)
- Password: `ontimeai2026` (env `API_PASSWORD`)
- Expiración: 8 horas

El middleware `AuthMiddleware` valida el token en **todos** los endpoints excepto `/auth/login`, `/docs`, `/openapi.json`.

**NO existe función `_require_auth()` en el código** — la validación es global vía middleware. No agregarla a nuevos endpoints.

### Endpoints

```
POST /auth/login                         → { access_token, token_type }
GET  /auth/me                            → { username }

GET  /flights                            → Flight[] (vuelos de hoy + última predicción)
GET  /flights/{fa_flight_id}             → Flight + shap[]
GET  /flight-history/{fa_flight_id}      → PredictionPoint[] (historial completo)

GET  /metrics/summary                    → KPI cards del dashboard
GET  /metrics/hourly                     → HourlyBucket[] (predicciones por hora)
GET  /metrics/model                      → modelo activo + AUC/Brier live
GET  /metrics/routes                     → RouteMetric[] (puntualidad por ruta)
GET  /metrics/routes/{origin}/{dest}/history → RouteHistoryPoint[] (serie diaria)

GET  /weather/{airport_code}             → observación METAR más reciente de la DB
GET  /operations/{airport_code}          → stats operacionales del día

GET  /test-cases                         → CP-01, CP-02 (casos de prueba formales)
```

**Variables de entorno del backend:**
- `ACTIVE_MODEL` — nombre del artefacto a cargar (default: `4year_v9`)
- `GCS_BUCKET` — nombre del bucket GCS (default vacío = usa DB local bundleada)
- `API_USERNAME` — usuario (default: `admin`)
- `API_PASSWORD` — contraseña (default: `ontimeai2026`)
- `JWT_SECRET_KEY` — clave JWT (default: `ontimeai-dev-secret-change-in-prod-32chars`)

---

## 6. Frontend (Next.js 16 App Router)

**URL**: `https://ontimeai-frontend-hq7henvhjq-uc.a.run.app`
**Framework**: Next.js 16.2.4 con Turbopack. React 19. Tailwind v4. shadcn/ui.

### Páginas y estado actual

| Ruta | Tipo | Descripción |
|---|---|---|
| `/` | Server | Dashboard: métricas, gráfico horario, tabla de vuelos, clima |
| `/flights` | Server | Buscador de vuelos con filtros por riesgo y búsqueda de texto |
| `/flights/[id]` | Server | Detalle: SHAP, historial de predicciones, resultado real vs predicho |
| `/routes` | Client | Puntualidad histórica por ruta (datos reales de actuals) |
| `/weather` | Server | Clima ATL: METAR real de IEM + gráfico horario |
| `/settings` | Client | Tema y perfil (local, sin backend) |
| `/login` | Client | Auth JWT |
| `/tesis/casos-de-prueba` | Client | CP-01, CP-02, CP-03 (validación formal) |
| `/tesis/pruebas` | Client | Sandbox de endpoints API |
| `/tesis/flights` | Client | Radar predictivo (experimental, oculto) |
| `/tesis/weather` | Client | Mapa METAR AWC/NOAA (experimental, oculto) |

**No existe `/reports`** — fue eliminada del sidebar y no hay página funcional.

### Reglas críticas de Next.js 16

1. **`useSearchParams()` siempre necesita `<Suspense>`** en la página que renderiza el componente que lo usa. `FlightsTable` usa `useSearchParams()` → todas las páginas que incluyan `<FlightsTable>` deben envolverlo en `<Suspense>`.

2. **Server Components async** — `WeatherCard` es un Server Component async. Debe estar dentro de `<Suspense>` en todas las páginas donde se usa.

3. **`localStorage` no disponible en Server Components** — `getAuthHeaders()` en `api.ts` maneja ambos contextos: en cliente usa `getToken()` (localStorage), en servidor usa `next/headers` cookies.

4. **`useSearchParams()` + `usePathname()`** — en `FlightsTable` se usa `usePathname()` para sincronizar filtros a la URL actual. **No hardcodear `/flights`** en ese efecto.

### cliente HTTP (api.ts)

```typescript
// Todos los métodos disponibles:
api.flights()                          // Flight[]
api.flight(id)                         // Flight
api.flightHistory(id)                  // PredictionPoint[]
api.weather(airportCode)               // WeatherData
api.routes()                           // RouteMetric[]
api.routeHistory(origin, dest)         // RouteHistoryPoint[]
api.summary()                          // MetricsSummary
api.hourly()                           // HourlyBucket[]
api.model()                            // ModelInfo
api.testCases()                        // TestCasesResponse
```

Todos usan `cache: "no-store"` — no hay caché del navegador ni de Next.js.

---

## 7. Despliegue

### Comandos

```bash
# Desde /Users/lologalaverna/Projects/Tesis/
./deploy.sh              # deploy completo (backend + frontend + job + scheduler)
./deploy.sh --backend    # solo backend
./deploy.sh --frontend   # solo frontend
./deploy.sh --job        # solo Cloud Run Job + scheduler
```

Requiere `gcloud` autenticado con el proyecto `ontimeai`:
```bash
gcloud auth login
gcloud config set project ontimeai
```

### Qué hace deploy.sh

1. Habilita APIs de GCP necesarias
2. Crea el bucket GCS si no existe
3. **Backend**: `gcloud builds submit` con `Dockerfile` → deploy en Cloud Run
4. **Job**: `gcloud builds submit` con `cloudbuild-job.yaml` + `Dockerfile.job` → `gcloud run jobs deploy`
5. **Scheduler**: crea o actualiza `ontimeai-pull-scheduler` con cron `*/30 * * * *`
6. **Frontend**: `gcloud builds submit` con `cloudbuild.yaml` pasando `NEXT_PUBLIC_API_URL` como substitution → deploy en Cloud Run

### Diagnóstico rápido

```bash
# Ver logs del último ciclo del job
gcloud run jobs executions list --job ontimeai-live-pull --region us-central1 --project ontimeai

# Ver logs de una ejecución específica
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=ontimeai-live-pull" \
  --project ontimeai --limit 100 --format "value(textPayload)"

# Cuántos ciclos corrieron
gcloud run jobs describe ontimeai-live-pull --region us-central1 --project ontimeai

# Estado de los servicios
gcloud run services list --platform managed --region us-central1 --project ontimeai

# Verificar DB en GCS
gsutil ls -l gs://ontimeai-live-db/

# Descargar DB para inspección local
gsutil cp gs://ontimeai-live-db/live_data.db /tmp/live_data.db
sqlite3 /tmp/live_data.db "SELECT COUNT(*) FROM predictions; SELECT COUNT(*) FROM actuals;"

# Forzar ejecución del job manualmente
gcloud run jobs execute ontimeai-live-pull --region us-central1 --project ontimeai
```

---

## 8. Datos mockeados / estado de completitud

### Todo real (100% datos del backend)
- Dashboard, vuelos, detalle de vuelo, meteorología, rutas, historial de predicciones, resultado real vs predicho

### Experimental / oculto (no en navegación principal)
- `/tesis/flights` — radar de vuelos con posiciones simuladas (marcado como "Simulado")
- `/tesis/weather` — mapa METAR con datos reales de AWC/NOAA (funcional pero fuera del MVP)
- `/tesis/casos-de-prueba` — validación formal CP-01/CP-02/CP-03 con datos reales
- `/tesis/pruebas` — sandbox de endpoints

### Eliminado
- `/reports` — quitado del sidebar, la página existe en el FS pero no se navega a ella

---

## 9. Bugs conocidos y fixes aplicados

### pandas 3.0 (en live.py)
- `pd.to_datetime()` con strings ISO8601+offset — usar `format="ISO8601"` explícitamente
- `pd.merge_asof()` rechaza null keys — filtrar nulls antes, concat después
- `_np.nan` en except block cuando `import numpy as _np` está en el try — usar `float("nan")`

### Next.js 16
- `useSearchParams()` sin Suspense bloquea static generation — toda página que renderice un componente con `useSearchParams()` necesita `<Suspense>`
- `localStorage` en Server Components lanza ReferenceError — `getAuthHeaders()` detecta contexto con `typeof window !== "undefined"`

---

## 10. Modelo ML — referencia rápida

| Atributo | Valor |
|---|---|
| Algoritmo | LightGBM (binary classification) |
| Artefacto activo | `4year_v9` (sin calibrar) / `4year_v9_recal` (calibrado) |
| Features | ~69 (v9 incluye BEARING_DEG, AIRCRAFT_FAMILY, PageRank, GDP_FLAG) |
| Target | `arr_delay_min > 15` → 1 (demorado), 0 (a tiempo) |
| Entrenamiento | BTS full-US 2021-2024 (~4 años, ~10M filas) |
| AUC test | 0.847 |
| AUC live | ~0.718 (distributional shift: ATL-only vs full-US) |
| Brier score live | ~0.143 |
| Calibración | IsotonicRegression sobre 2,266 actuals live (v9_recal, 2026-05-12) |
| Umbral | `quantile@0.22` recalculado cada ciclo sobre el batch target |
| Actuals acumulados | >12,000 (desde 2026-05-19) |

Para cambiar el modelo activo sin redesplegar: modificar `ACTIVE_MODEL` en las variables de entorno del servicio Cloud Run backend. Opciones válidas: cualquier carpeta en `artifacts/` que contenga el booster.

---

## 11. Convenciones de desarrollo

- **No iniciar servidores locales** (dev server, uvicorn local) — el usuario tiene una PC con recursos limitados
- **Deploy siempre vía `./deploy.sh`** con `run_in_background: true` en la herramienta Bash
- **Commits**: no hacer sin instrucción explícita del usuario
- **No hay `_require_auth(request)` helper** en api.py — el middleware global maneja auth en todos los endpoints
- **Fechas en fmtTime**: mostrar siempre `dd/MM HH:mm` (día/mes hora:minuto UTC) — nunca solo hora
- **Suspense obligatorio** para: `<WeatherCard />`, `<FlightsTable />` en cualquier página nueva
