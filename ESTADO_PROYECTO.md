# OnTimeAI — Estado del Proyecto (17 mayo 2026)

## Qué es esto

Sistema de predicción de retrasos de vuelos en tiempo real para el aeropuerto ATL (Hartsfield-Jackson Atlanta). Modelo LightGBM entrenado sobre datos BTS 2021-2024, con pipeline de features en vivo vía AeroAPI + IEM (clima).

---

## Stack

```
OnTimeAI-Backend/   ← Python: modelo, pipeline live, API REST
OnTimeAI-Frontend/  ← Next.js: dashboard visual
```

### Backend
- `live_pull.py` — cron principal: descarga vuelos de AeroAPI, construye features, predice y guarda en `live_data.db`
- `api.py` — FastAPI: sirve predicciones y métricas al frontend
- `eval_live.py` — evalúa métricas del modelo sobre actuals ya guardados
- `recalibrate_live.py` — recalibra el modelo sobre datos live (usa sigmoid, no isotonic)
- `ontimeai/` — librería interna: features, calibración, SHAP, lineage

### Frontend
- `src/lib/api.ts` — cliente tipado que consume el backend FastAPI
- `src/components/` — todos los componentes consumen datos reales (no mocks)
- `.env.local` → `NEXT_PUBLIC_API_URL=http://localhost:8000`

---

## Modelo activo: v9

**Artefacto:** `artifacts/4year_v9`

| Versión | Entrenado sobre | AUC test | AUC live |
|---------|----------------|----------|----------|
| v7_recal | BTS multi-aeropuerto | — | 0.681 |
| **v9** | BTS 2021-2024 ATL, 27.5M filas | 0.8602 | **0.714** |
| v9_recal | Isotonic sobre live (descartado) | — | 0.661 ← destruyó AUC |

**v9 ganó sobre v7_recal:**
- ΔAUC = **+0.033**
- ΔBrier = **−0.105** (−62%)

El v9_recal con isotonic se descartó porque con <2k muestras isotonic colapsa scores distintos al mismo valor. Si se quiere recalibrar usar `--method sigmoid`.

---

## Resultados finales — eval al 17/05/2026

Guardados en: `logs/live_eval_FINAL_v9.json`

```
v7_recal:  n=8569   AUC=0.681  Brier=0.168  actual_rate=21.3%
v9:        n=2117   AUC=0.714  Brier=0.063  actual_rate=7.4%
v9_recal:  n=176    AUC=0.661  (descartado — isotonic overfit)
```

Período live v9: 15 mayo 2026 → 17 mayo 2026 (3 días, datos reales de KATL)

---

## Por qué se pausó el sistema

La API key de AeroAPI (FlightAware) acumuló **$50.53 USD** en uso (11,130 llamadas) sin tarjeta de crédito cargada. Se pausó el cron para no sumar más deuda.

**Estado actual:**
- Cron `live_pull.py`: **DETENIDO**
- Cron `eval_live.py`: activo pero sin datos nuevos
- Base de datos: `live_data.db` con 38,564 predicciones (4 mayo → 17 mayo)

---

## Cómo retomar con nueva API key

### 1. Crear cuenta nueva en FlightAware y obtener key

En [flightaware.com/aeroapi/portal](https://flightaware.com/aeroapi/portal). El plan gratuito da 500 queries/mes sin tarjeta.

### 2. Actualizar la key

```bash
# Editar .env
echo "AEROAPI_KEY=tu_nueva_key_aqui" > .env
```

### 3. Activar el cron (cada 30 min ≈ $1.50/día)

```bash
(crontab -l; echo "*/30 * * * * cd /Users/lologalaverna/Projects/Tesis/OnTimeAI-Backend && /bin/zsh -c 'source .venv/bin/activate && python3 live_pull.py >> logs/live_v7.log 2>&1'") | crontab -
```

### 4. Verificar que funciona

```bash
cd OnTimeAI-Backend
source .venv/bin/activate
python3 live_pull.py   # correr manualmente una vez
tail -f logs/live_v7.log
```

---

## Cómo levantar el sistema completo

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

Modelos disponibles: `4year_v9`, `4year_v9_recal`, `4year_v7_recal`

### Frontend (Next.js)

```bash
cd OnTimeAI-Frontend
npm run dev   # http://localhost:3000
```

### Endpoints del backend

| Endpoint | Descripción |
|----------|-------------|
| `GET /flights` | Lista de vuelos del día con predicciones |
| `GET /flights/{id}` | Detalle de un vuelo + SHAP |
| `GET /metrics/summary` | Métricas resumen (AUC, Brier, total vuelos) |
| `GET /metrics/hourly` | Probabilidad promedio por hora |
| `GET /metrics/model` | Info del modelo activo |

---

## Recalibrar con datos nuevos (cuando haya suficientes actuals)

```bash
cd OnTimeAI-Backend
source .venv/bin/activate

# Recalibrar v9 con datos live acumulados (usar sigmoid, nunca isotonic)
python3 recalibrate_live.py \
  --source-artifact artifacts/4year_v9 \
  --out artifacts/4year_v9_recal \
  --since "2026-05-15 01:43" \
  --method sigmoid

# Evaluar el nuevo artefacto
python3 eval_live.py --out logs/eval_post_recal.json
```

Regla: esperar al menos **500 actuals únicos** antes de recalibrar. Con menos el calibrador no generaliza.

---

## Estructura de la base de datos (`live_data.db`)

Dos tablas principales:

- **`predictions`** — una fila por (fa_flight_id, predicted_at_utc): probabilidad predicha, features usadas
- **`actuals`** — resultado real del vuelo: si llegó retrasado y por cuántos minutos

`eval_live.py` une ambas por `fa_flight_id`, toma la última predicción por vuelo (deduplicación) y calcula AUC/Brier.

---

## Modelo activo en el cron

`live_pull.py` usa por defecto `artifacts/4year_v9`. Para cambiar:

```python
# live_pull.py línea ~30
ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "4year_v9"
```

---

## Versiones del modelo — historial

```
v7_recal   2000-01-01  ← baseline, multi-aeropuerto BTS
v9         2026-05-15  ← modelo actual, ATL 2021-2024
v9_recal   2026-05-16  ← descartado (isotonic overfit)
v9         2026-05-17  ← rollback a v9 limpio
```

El script `eval_live.py` etiqueta cada predicción según estas fechas para comparar modelos correctamente.
