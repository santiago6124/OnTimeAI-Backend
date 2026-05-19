# OnTimeAI — Estado del Proyecto (18 mayo 2026)

## Qué es esto

Sistema de predicción de retrasos de vuelos en tiempo real para el aeropuerto ATL (Hartsfield-Jackson Atlanta). Modelo LightGBM entrenado sobre datos BTS 2021-2024, con pipeline de features en vivo vía AeroAPI + IEM (clima). Tesis UCC Grupo 9.

**Integrantes:** Santiago Carranza · Lorenzo Galaverna · Facundo Oliva Marchetto · Mateo Pappalardo

---

## URLs en producción

| Servicio | URL |
|---------|-----|
| Frontend | https://ontimeai-frontend-hq7henvhjq-uc.a.run.app |
| Backend API | https://ontimeai-backend-hq7henvhjq-uc.a.run.app |
| Swagger docs | https://ontimeai-backend-hq7henvhjq-uc.a.run.app/docs |
| DB bucket | gs://ontimeai-live-db/live_data.db |

---

## Stack en producción

```
Cloud Scheduler (*/30 min)
  → Cloud Run Job: ontimeai-live-pull
      → descarga live_data.db desde GCS
      → live_pull.py (AeroAPI → features → LightGBM → SQLite)
      → sube live_data.db actualizada a GCS
          ↑
Cloud Run: ontimeai-backend (FastAPI)
      → descarga DB de GCS en cold start
      → sirve /flights, /metrics, etc.

Cloud Run: ontimeai-frontend (Next.js)
      → consume el backend
```

---

## Modelo activo: v9

**Artefacto:** `artifacts/4year_v9`

| Versión | Entrenado sobre | AUC test | AUC live |
|---------|----------------|----------|----------|
| v7_recal | BTS multi-aeropuerto | — | 0.681 |
| **v9** | BTS 2021-2024 ATL, 27.5M filas | 0.8602 | **0.714** |
| v9_recal | Isotonic sobre live (descartado) | — | 0.661 ← overfit |

Resultados finales: `logs/live_eval_FINAL_v9.json`  
Live period: 4 mayo → 17 mayo 2026 · 38,564 predicciones · 8,569 actuals.

**Regla de recalibración:** mínimo 500 actuals antes de recalibrar. Usar siempre `--method sigmoid`, nunca isotonic con poco dato.

---

## Cómo redesployar

```bash
cd /Users/lologalaverna/Projects/Tesis

./deploy.sh              # deploy completo
./deploy.sh --backend    # solo backend
./deploy.sh --frontend   # solo frontend
./deploy.sh --job        # solo cloud run job + scheduler
```

Requiere `gcloud` autenticado: `gcloud auth login && gcloud config set project ontimeai`

Documentación completa: `docs/05_infraestructura_gcp.md`

---

## Cómo verificar que el pipeline live funciona

```bash
# Disparar el job manualmente (en lugar de esperar el scheduler)
gcloud run jobs execute ontimeai-live-pull \
  --region=us-central1 --project=ontimeai --wait

# Ver ejecuciones
gcloud run jobs executions list \
  --job=ontimeai-live-pull --region=us-central1 --project=ontimeai

# Logs del backend
gcloud run services logs read ontimeai-backend \
  --region=us-central1 --project=ontimeai --limit=50
```

---

## Cómo correr localmente

### Backend (FastAPI)

```bash
cd OnTimeAI-Backend
source .venv/bin/activate
uvicorn api:app --reload --port 8000
```

Para cambiar el modelo activo sin tocar código:
```bash
ACTIVE_MODEL=4year_v9 uvicorn api:app --reload --port 8000
```

### Frontend (Next.js)

```bash
cd OnTimeAI-Frontend
# Asegurar que .env.local tiene NEXT_PUBLIC_API_URL=http://localhost:8000
pnpm dev   # http://localhost:3000
```

### Live pull local (sin GCS, usa live_data.db local)

```bash
cd OnTimeAI-Backend
source .venv/bin/activate
AEROAPI_KEY=tu_key python3 live_pull.py
```

---

## Estructura de la base de datos (`live_data.db`)

Dos tablas principales:

- **`predictions`** — una fila por `(fa_flight_id, predicted_at_utc)`: probabilidad predicha, features usadas
- **`actuals`** — resultado real del vuelo: si llegó retrasado y cuántos minutos

`eval_live.py` une ambas por `fa_flight_id`, toma la última predicción por vuelo y calcula AUC/Brier.

```bash
# Evaluar el modelo actual sobre todos los actuals
python3 eval_live.py --out logs/eval_$(date +%F).json
```

---

## Recalibrar con datos nuevos

```bash
cd OnTimeAI-Backend
source .venv/bin/activate

python3 recalibrate_live.py \
  --source-artifact artifacts/4year_v9 \
  --out artifacts/4year_v9_recal \
  --since "2026-05-18" \
  --method sigmoid

python3 eval_live.py --out logs/eval_post_recal.json
```

Esperar al menos **500 actuals únicos** antes de recalibrar.

---

## Poblar DB con datos demo (para sandbox/pruebas)

```bash
cd OnTimeAI-Backend
source .venv/bin/activate
python3 seed_demo.py           # inserta 40 vuelos de hoy con predicciones + actuals
python3 seed_demo.py --clear   # limpia la tabla y re-siembra
```

---

## Endpoints del backend

| Endpoint | Descripción |
|----------|-------------|
| `GET /flights` | Lista de vuelos del día con predicciones |
| `GET /flights/{id}` | Detalle de un vuelo + SHAP values |
| `GET /metrics/summary` | AUC, Brier, total predicciones |
| `GET /metrics/hourly` | Probabilidad promedio por hora |
| `GET /metrics/model` | Info del modelo activo |
| `GET /docs` | Swagger UI |

---

## Versiones del modelo — historial

```
v7_recal   antes 2026-05-15  ← baseline, multi-aeropuerto BTS
v9         2026-05-15        ← modelo actual, ATL 2021-2024
v9_recal   2026-05-16        ← descartado (isotonic overfit con <200 muestras)
v9         2026-05-17        ← rollback a v9 limpio (modelo en producción)
```
