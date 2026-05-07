# OnTimeAI — Backend

**Predicción de retrasos aéreos y modelado de efecto cascada mediante Machine Learning.**

Proyecto de tesis. Backend con pipeline de ML basado en LightGBM para clasificar la probabilidad de retraso de vuelos individuales en el aeropuerto de Atlanta (ATL) a partir de datos operacionales (BTS) y meteorológicos (IEM / NOAA ASOS METAR).

Integrantes: Santiago Carranza · Lorenzo Galaverna · Facundo Oliva Marchetto · Mateo Pappalardo.

---

## Tabla de contenidos

1. [Estado del proyecto](#estado-del-proyecto)
2. [Quickstart](#quickstart)
3. [Estructura del repositorio](#estructura-del-repositorio)
4. [Dataset maestro](#dataset-maestro)
5. [Pipeline de entrenamiento](#pipeline-de-entrenamiento)
6. [Feature engineering](#feature-engineering)
7. [Disciplina anti-leakage](#disciplina-anti-leakage)
8. [Resultados](#resultados-4-años-753-k-vuelos-target-binario-delay--15-min)
9. [Modelo: LightGBM (justificación)](#modelo-lightgbm-justificación)
10. [Explicabilidad (SHAP)](#explicabilidad-shap)
11. [Métricas y evaluación](#métricas-y-evaluación)
12. [Tests](#tests)
13. [Artifacts y serialización](#artifacts-y-serialización)
14. [CLI de referencia](#cli-de-referencia)
15. [Efecto cascada (TODO post-MVP)](#efecto-cascada-todo-post-mvp)
16. [Instalación](#instalación)
17. [Referencias académicas](#referencias-académicas)

---

## Estado del proyecto

| Componente | Estado | Notas |
|---|---|---|
| Pipeline de datos (BTS + IEM merge) | Listo | `construir_dataset_maestro_multi.py` — multi-año |
| Descarga BTS 2022-2024 | Listo | `descarga_bts.py` (PREZIP, 36 meses) |
| Descarga IEM 2022-2024 | Listo | `descarga_data_iem_multi.py` (16 aeropuertos × N años) |
| Dataset maestro ATL 2022-2025 | Listo | **753 K vuelos · 65 cols · 4 años · cobertura meteo 99.99 %** |
| Feature engineering base | Listo | cyclical, congestión, interacciones meteo |
| **Lineage features** (leakage-safe) | Listo | `ontimeai/lineage.py` — tail, carrier, origin |
| **Rolling window features** | Listo | carrier 24h/7d · origin 1h/6h/24h |
| **Calibración isotónica** | Listo | `ontimeai/calibration.py` |
| **Hyperparameter tuning (Optuna)** | Listo | `tune.py` con TPE sampler |
| Clasificador individual (MVP) | **Listo · AUC 0.813, Acc 78.2 %** | LightGBM binario / multiclase |
| Tests unitarios e integración | **Listo** | **58 tests** · `pytest` |
| Explicabilidad SHAP | Listo | TreeExplainer + importancia global |
| Serialización de artifacts | Listo | `model.lgb` + `meta.joblib` + `metrics.json` + calibrador |
| **Efecto cascada** | **TODO** | Stub con roadmap en `ontimeai/cascade.py` |
| API REST (FastAPI) | Pendiente | Consumir los artifacts generados aquí |
| Dashboard (Next.js) | Pendiente | Repo separado |
| Streaming tiempo real | Pendiente | Post-MVP |

---

## Quickstart

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. (Opcional) Descargar más años de datos BTS + IEM
python3 descarga_bts.py 2022 2023 2024
python3 descarga_data_iem_multi.py 2022 2023 2024
python3 construir_dataset_maestro_multi.py 2022 2023 2024
python3 concatenar_masters.py dataset_maestro_ATL_2022-2024_*.csv dataset_maestro_ATL_2025_*.csv

# 3. (Opcional) Tuning de hiperparámetros con Optuna
python3 tune.py --n-trials 25 --no-balance --out artifacts/best_params.json

# 4. Entrenar con el stack completo (lineage + rolling + calibración + params óptimos)
python3 train.py --num-boost-round 3000 --early-stopping 150 --no-balance \
    --best-params artifacts/best_params.json --artifacts ./artifacts/v3

# 5. Entrenar en modo smoke (subsample rápido)
python3 train.py --subsample 50000 --num-boost-round 300 --artifacts ./artifacts/smoke

# 6. Entrenar target multiclase (C0/C1/C2/C3)
python3 train.py --target multiclass

# 7. Predecir sobre un CSV con el schema maestro
python3 predict.py --artifact ./artifacts/v3 --input vuelos_nuevos.csv --output preds.csv

# 8. Ejecutar los tests
python3 -m pytest
```

---

## Estructura del repositorio

```
OnTimeAI-Backend/
├── AnteProyecto-OnTimeAI.md          # Anteproyecto académico
├── construir_dataset_maestro_atl_2025.py  # Pipeline de construcción del dataset (BTS + IEM)
├── descarga_data_iem.py              # Descarga de METAR desde Iowa Environmental Mesonet
├── clima_iem_asos_2025_utc.csv       # Datos METAR 2025 (UTC) para los aeropuertos del alcance
├── dataset_maestro_ATL_2025_BTS_IEM_ORIG_DEST.csv  # Dataset maestro (target de entrenamiento)
├── EDA_*.ipynb                       # Notebooks de análisis exploratorio
├── baseline_epic1_imbalance_rescue.ipynb   # Baseline exploratorio (Épica 1)
│
├── ontimeai/                         # Paquete principal
│   ├── config.py                     # TrainConfig + constantes (columnas leaky, categoricals, bins)
│   ├── data.py                       # Carga + filtros anti-leakage
│   ├── features.py                   # Feature engineering base (cyclical, congestión, interacciones)
│   ├── lineage.py                    # Lineage + rolling-window features (leakage-safe)
│   ├── calibration.py                # Isotonic / Platt scaling para probabilidades
│   ├── tuning.py                     # Hyperparameter search con Optuna TPE
│   ├── split.py                      # Split temporal 60/20/20
│   ├── model.py                      # Entrenamiento LightGBM + threshold tuning + serialización
│   ├── evaluation.py                 # Métricas binarias y multiclase
│   ├── explainability.py             # Wrappers sobre SHAP TreeExplainer
│   ├── pipeline.py                   # Orquestación end-to-end
│   └── cascade.py                    # TODO: efecto cascada (LSTM/GNN)
│
├── tests/                            # Suite pytest (36 tests)
│   ├── conftest.py                   # Fixture con dataset sintético
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_split.py
│   ├── test_model.py
│   ├── test_evaluation.py
│   ├── test_pipeline.py
│   └── test_explainability.py
│
├── descarga_bts.py                   # Downloader BTS PREZIP multi-año
├── descarga_data_iem_multi.py        # Downloader IEM ASOS multi-año
├── construir_dataset_maestro_multi.py  # Builder multi-año (PREZIP + legacy schema)
├── concatenar_masters.py             # Concat de masters anuales con dedup
├── train.py                          # CLI de entrenamiento
├── tune.py                           # CLI de Optuna hyperparameter tuning
├── predict.py                        # CLI de inferencia
├── artifacts/                        # Modelos serializados + best_params.json
├── data_raw/                         # BTS zips extraídos + IEM yearly CSVs (no-commit)
├── requirements.txt
└── pyproject.toml                    # Config de pytest
```

---

## Dataset maestro

**`dataset_maestro_ATL_2022-2025_BTS_IEM_ORIG_DEST.csv`** — **753 060 filas × 65 columnas, 4 años (2022-2025)**, centrado en ATL con 16 aeropuertos conectados (ATL, LGA, MCO, FLL, MIA, DFW, DCA, EWR, TPA, ORD, PHL, DEN, LAX, BWI, LAS, BOS).

Desglose por año: 2022 (178 K) · 2023 (196 K) · 2024 (193 K) · 2025 (186 K). Cobertura meteorológica ≥ 99.98 % origen y destino.

Construido por `construir_dataset_maestro_atl_2025.py`, que hace:

1. Lee los CSV mensuales de BTS 2025 filtrando vuelos en los que ATL es origen o destino.
2. Convierte `FL_DATE` + `CRS_DEP_TIME` local → UTC por zona horaria del aeropuerto.
3. Hace `merge_asof` con datos METAR IEM por estación, tolerancia de 90 min, dirección `nearest` — tanto para meteo de origen (`ORIG_WX_*`) como destino (`DEST_WX_*`).
4. Marca flags derivados meteo: `PRECIP_FLAG` (p01m > 0), `LOW_VIS_FLAG` (vsby < 3), `STRONG_WIND_FLAG` (sknt ≥ 20 o gust ≥ 30).

### Grupos de columnas

| Grupo | Columnas clave | Uso |
|---|---|---|
| Identificación del vuelo | `FL_DATE`, `OP_CARRIER`, `TAIL_NUM`, `ORIGIN`, `DEST`, `OP_CARRIER_FL_NUM` | Categóricas / ID |
| Schedule | `CRS_DEP_TIME`, `CRS_ARR_TIME`, `CRS_DEP_MIN`, `CRS_ELAPSED_TIME`, `DISTANCE` | Features |
| Derivadas | `FLOW_ATL` (DEP/ARR), `PAR_AIRPORT` (par conectado) | Features |
| Meteo origen | `ORIG_WX_TMPC`, `ORIG_WX_DWPC`, `ORIG_WX_RELH`, `ORIG_WX_DRCT`, `ORIG_WX_SKNT`, `ORIG_WX_ALTI`, `ORIG_WX_P01M`, `ORIG_WX_VSBY`, `ORIG_WX_GUST`, `ORIG_WX_CODES`, flags, `ORIG_WX_MATCH_GAP_MIN` | Features |
| Meteo destino | `DEST_WX_*` (mismo suite) | Features |
| **Post-hoc (leakage)** | `DEP_TIME`, `ARR_TIME`, `DEP_DELAY`, `ARR_DELAY`, `ACTUAL_ELAPSED_TIME`, `AIR_TIME`, `CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY` | **Target (ARR_DELAY) o removidas** |
| Flags de filtro | `CANCELLED`, `DIVERTED` | Excluidos del entrenamiento |

### Tipo de predicción: Clasificación supervisada (no regresión)

El modelo realiza **clasificación supervisada**, no regresión. En lugar de predecir los minutos exactos de retraso (regresión), el modelo predice la **probabilidad de que un vuelo pertenezca a una categoría de retraso**.

| Aspecto | Clasificación (lo que usamos) | Regresión (alternativa descartada) |
|---|---|---|
| **Output** | Probabilidad de retraso (0.0 – 1.0) + etiqueta binaria/multiclase | Minutos de retraso estimados |
| **Función de pérdida** | Binary log-loss / cross-entropy | MSE / MAE |
| **Métrica principal** | ROC AUC (capacidad de ranking) | RMSE / MAE |
| **Justificación** | Alineado con el estándar FAA de "on-time" (≤ 15 min); la decisión operacional es binaria (retraso sí/no); más robusto ante outliers extremos (vuelos con +300 min de delay) | Requeriría modelar la distribución completa de delays, muy sesgada (heavy-tail) |

**¿Por qué clasificación y no regresión?**
1. La distribución de `ARR_DELAY` es altamente sesgada (heavy-tailed): la mayoría de los vuelos llegan a tiempo, pero los outliers tienen +300 min de delay. La regresión se distorsiona con estos extremos.
2. La decisión operacional es fundamentalmente categórica: un pasajero necesita saber "¿se retrasa o no?", no "se retrasará 17.3 minutos".
3. El output probabilístico (`proba_delay`) ya codifica la incertidumbre del modelo — funciona como un "termómetro" continuo entre 0 y 1.

### Target

Configurable via `TrainConfig.target`:

- **`binary`** (default): `ARR_DELAY > 15` → `{0, 1}`. Alineado con el estándar FAA de "on-time" y con el objetivo académico del 75% de accuracy.
- **`multiclass`**: bins `[-∞, 15, 30, 60, ∞]` → `{C0, C1, C2, C3}`. Aporta granularidad operacional para aerolíneas.

---

## Pipeline de entrenamiento

Orquestado por `ontimeai.pipeline.run_training(cfg)`. Etapas:

1. **Carga** del CSV maestro con `parse_dates=["FL_DATE"]`.
2. **Filtro de vuelos válidos**: excluye `CANCELLED=1`, `DIVERTED=1`, y filas con `ARR_DELAY` nulo.
3. **Construcción del target** (`TARGET`) a partir de `ARR_DELAY`.
4. **Drop de columnas leaky** (ver próxima sección).
5. **Normalización de flags meteorológicos** (`True`/`False` strings → `int8`).
6. **Feature engineering**:
   - Cyclical: `sin/cos` de hora del día, día de semana, mes.
   - Congestión: conteo de vuelos programados en ±30 min en la misma fecha y aeropuerto (origen y destino).
   - Interacciones meteo: AND entre flags de origen y destino (`wx_both_precip`, `wx_both_low_vis`, `wx_both_strong_wind`).
7. **Split temporal 60/20/20** ordenado por `FL_DATE + CRS_DEP_MIN`. *Nunca* random (leakage temporal).
8. **Construcción de la matriz de features**: drop de IDs + datetimes redundantes, cast a `category` de las columnas de alta cardinalidad, serialización del mapping de categorías.
9. **Entrenamiento LightGBM** con `sample_weight` balanceado, early stopping sobre val.
10. **Threshold tuning** sobre val (F1 por default, Youden opcional).
11. **Evaluación** sobre val y test.
12. **Serialización** de artifact.

---

## Feature engineering

### Features directas (después del filtrado)

| Columna | Tipo | Fuente |
|---|---|---|
| `MONTH`, `DAY_OF_MONTH`, `DAY_OF_WEEK` | int | Schedule |
| `CRS_DEP_MIN`, `CRS_ELAPSED_TIME`, `DISTANCE` | float | Schedule |
| `OP_CARRIER`, `TAIL_NUM`, `ORIGIN`, `DEST`, `PAR_AIRPORT`, `FLOW_ATL` | category | Schedule + derivadas |
| `ORIG_WX_*`, `DEST_WX_*` (numérico y flags) | float / int8 | IEM METAR |
| `ORIG_WX_CODES`, `DEST_WX_CODES` | category | IEM (wxcodes) |
| `*_WX_MATCH_GAP_MIN` | float | Calidad del match meteo |

### Features engineered base

| Columna | Descripción | Justificación |
|---|---|---|
| `dep_hour_sin`, `dep_hour_cos` | Codificación cíclica del minuto programado de salida | Evita el salto 23:59 ↔ 00:00 |
| `dep_dow_sin`, `dep_dow_cos` | Cyclical day-of-week | Ídem para el lunes↔domingo |
| `dep_month_sin`, `dep_month_cos` | Cyclical mes | Estacionalidad anual |
| `congestion_orig_window` | Vuelos programados en ±30 min en ORIGIN el mismo día | Proxy de congestión operacional (correlaciona con `NAS_DELAY`, que es leaky) |
| `congestion_dest_window` | Ídem para DEST | Ídem |
| `wx_both_precip` | Lluvia en origen **y** destino | Efectos meteorológicos combinados |
| `wx_both_low_vis` | Baja visibilidad en ambos extremos | IFR en origen y destino |
| `wx_both_strong_wind` | Viento/ráfagas en ambos extremos | Impacto combinado |

### Lineage features (`ontimeai/lineage.py`) — leakage-safe

Observabilidad estricta: un vuelo prior solo es usable si su **actual arrival time** (CRS_ARR + ARR_DELAY) es ≤ `CRS_DEP` del vuelo actual.

| Columna | Ventana | Justificación |
|---|---|---|
| `prev_arr_delay_tail` | Vuelo inmediato anterior del mismo `TAIL_NUM`, mismo día | Principal señal de cascade intra-aircraft |
| `prev_turnaround_tail_min` | Turnaround programado entre arr previo → dep actual | Tight turnaround = mayor riesgo |
| `tail_flights_today_prior` | Count acumulado de vuelos previos del tail ese día | Fatiga operacional intradía |
| `carrier_delay_rate_yday` | Share delayed del carrier el día previo | Daily lag — tendencia diaria |
| `origin_delay_rate_yday` | Share delayed en el origen el día previo | Daily lag — estado del aeropuerto |
| `carrier_delay_rate_24h` | Rolling 24h — share delayed del carrier | Estado reciente del carrier |
| `carrier_delay_rate_7d` | Rolling 7 días — share delayed del carrier | Baseline semanal del carrier |
| `origin_delay_rate_1h` | Rolling 1h — share delayed en el origen | Ola de delay corta |
| `origin_delay_rate_6h` | Rolling 6h — share delayed en el origen | Ventana media |
| `origin_delay_rate_24h` | Rolling 24h — share delayed en el origen | Estado del día |

Todas las features rolling se computan por grupo via `searchsorted` sobre timestamps de visibilidad ordenados — O(N log N) sobre 750 K filas (~10 s).

---

## Disciplina anti-leakage

El riesgo más grande del proyecto es entrenar con columnas que solo existen **después** de que el vuelo voló. Las columnas post-hoc removidas explícitamente son:

```
DEP_TIME, ARR_TIME, DEP_DELAY, DEP_DELAY_NEW, ARR_DELAY_NEW,
ACTUAL_ELAPSED_TIME, AIR_TIME,
CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY
```

`ARR_DELAY` se usa únicamente para construir el target y luego se dropea del feature matrix.

El test `tests/test_pipeline.py::test_prepare_dataset_has_no_leakage` **falla** si alguna columna leaky aparece en el feature matrix — es el guardrail principal del pipeline.

Split temporal estricto: **train** primero ≈60% del año, **val** ≈20% siguiente, **test** ≈20% final. Ordenado por `FL_DATE + CRS_DEP_MIN`. Replica cómo el modelo se usaría en producción: entrenado con pasado, predice futuro.

---

## Resultados (4 años, 753 K vuelos, target binario delay > 15 min)

Trayectoria de mejoras aplicadas incrementalmente sobre el conjunto de test (20 % final cronológico):

| Versión | Test Accuracy | Test ROC AUC | Test F1 | Test Brier |
|---|---|---|---|---|
| v0 smoke (60 K subsample) | 0.449 | 0.622 | 0.371 | 0.158 |
| v1 baseline (4 años full) | 0.674 | 0.708 | 0.494 | 0.174 |
| v2 + lineage + calibración | 0.771 | 0.796 | 0.582 | 0.141 |
| **v3 + rolling + Optuna** | **0.782** | **0.813** | **0.604** | **0.135** |

**Ganancias v0 → v3**: accuracy +33 pts, ROC AUC +19 pts, F1 +23 pts, Brier −0.023.

### Top-10 features por gain en v3

| # | Feature | % gain | Familia |
|---|---|---|---|
| 1 | `prev_arr_delay_tail` | 18.2 | Lineage |
| 2 | `carrier_delay_rate_24h` | 10.5 | Rolling |
| 3 | `prev_turnaround_tail_min` | 10.0 | Lineage |
| 4 | `CRS_DEP_MIN` | 5.9 | Schedule |
| 5 | `tail_flights_today_prior` | 5.8 | Lineage |
| 6 | `dep_hour_sin` | 4.0 | Cyclical |
| 7 | `ORIG_WX_CODES` | 3.9 | Weather |
| 8 | `TAIL_NUM` | 3.7 | Categorical |
| 9 | `PAR_AIRPORT` | 2.8 | Schedule |
| 10 | `origin_delay_rate_6h` | 2.8 | Rolling |

Las 8 features de lineage + rolling concentran **≈ 58 % del gain total** del modelo.

### Hiperparámetros óptimos (Optuna TPE, 25 trials)

```json
{
  "num_leaves": 218,
  "learning_rate": 0.0266,
  "min_data_in_leaf": 1876,
  "feature_fraction": 0.931,
  "bagging_fraction": 0.922,
  "bagging_freq": 8,
  "reg_alpha": 0.674,
  "reg_lambda": 0.003
}
```

## Modelo: LightGBM (justificación)

Elegido sobre alternativas (XGBoost, CatBoost, Random Forest, regresión logística) por:

| Criterio | Razón |
|---|---|
| Datos tabulares con mix numérico + categórico | GBDT domina benchmarks tabulares (Shwartz-Ziv & Armon 2022; Wandelt et al. 2025 revisión en flight delay) |
| Categóricas de alta cardinalidad (`TAIL_NUM`, `ORIGIN`) | LightGBM las maneja nativamente (dtype `category`), sin one-hot explosion |
| Faltantes meteo (`M`, gap > 90 min) | Los splits GBDT manejan `NaN` sin imputación arbitraria |
| 185K filas | Entrenamiento en <1 min CPU; sobredimensionado para deep learning |
| Explicabilidad SHAP requerida | `TreeExplainer` es exacto y ~100× más rápido que `KernelSHAP` |
| Imbalance de clase (C3, > 60 min) | `sample_weight='balanced'` + threshold tuning sobre F1 en val |
| Latencia API p95 < 500 ms | Predicción de un árbol < 5 ms |

**Hiperparámetros por default** (en `config.TrainConfig.lgb_params`):

```python
learning_rate    = 0.05
num_leaves       = 63
feature_fraction = 0.9
bagging_fraction = 0.8
bagging_freq     = 5
min_data_in_leaf = 200
num_boost_round  = 2000
early_stopping   = 100 rounds
```

Tuning exhaustivo de hiperparámetros (Optuna, grid search) queda pendiente para la Épica 2.

---

## Explicabilidad (SHAP)

Los wrappers en `ontimeai.explainability` permiten:

- **`compute_shap_values(booster, X)`**: computa SHAP values vía `TreeExplainer`.
- **`global_feature_importance(shap_values, feature_names) -> pd.Series`**: importancia global por feature (|SHAP| promedio), ordenada.
- **`explain_instance(shap_values, feature_names, row_idx, top_n=10) -> pd.DataFrame`**: top-N features que explican una predicción individual.

El soporte multiclass está incluido (lista de arrays SHAP por clase).

Uso típico en un notebook de análisis:

```python
from ontimeai.explainability import compute_shap_values, global_feature_importance
from ontimeai.model import load_artifact

meta = load_artifact("./artifacts")
# Asumiendo X de inferencia preparado con predict.prepare_inference_frame
sv = compute_shap_values(meta["booster"], X)
top = global_feature_importance(sv, meta["feature_cols"]).head(15)
print(top)
```

---

## Métricas y evaluación

### Binary target

- `accuracy`, `precision`, `recall`, `f1`
- `roc_auc`, `brier`, `log_loss`
- `support_pos`, `support_neg`
- Confusion matrix (2×2)

### Multiclass target (C0-C3)

- `accuracy`, `f1_macro`, `f1_per_class` (por clase C0/C1/C2/C3)
- `roc_auc_macro_ovr`, `log_loss`
- `per_class_report` (precision/recall/f1/support por clase)
- Confusion matrix (4×4)

Todas las métricas se escriben a `artifacts/metrics.json` al finalizar el entrenamiento.

---

## Tests

**58 tests en `tests/`**, todos verdes (`python3 -m pytest`).

| Suite | Cobertura |
|---|---|
| `test_data.py` | Filtro excluye cancelados/desviados/nulos · idempotencia · drop de columnas leaky · preserva schedule cols |
| `test_features.py` | Cyclical en círculo unitario · handling de NaN · window counts ±30 · congestión no cruza días · build_target binary y multiclass ordinal · round-trip de categorical mapping · normalización de flags |
| `test_split.py` | Sumas = N · sin overlap · orden cronológico estricto · rechaza fracciones inválidas |
| `test_model.py` | Shape de proba binaria · threshold tuning en rango válido · serialize → load → predict idéntico · predict_label binario y multiclase |
| `test_evaluation.py` | Predicción perfecta · valores random en rangos válidos · confusion matrix |
| `test_pipeline.py` | End-to-end binario y multiclase · guard de leakage · artifacts persistidos · cascade stub levanta NotImplementedError |
| `test_explainability.py` | SHAP importance positivo · explain_instance top-N |
| `test_lineage.py` | Tail lineage leakage-safe (observability verificada) · cumcount por día · NaN en primer vuelo · rolling windows per-grupo · no contaminación entre carriers |
| `test_calibration.py` | Isotonic + sigmoid fit · monotonía · handling de extremos 0/1 · preserva rango [0,1] |
| `test_bts_normalizer.py` | Standardize PREZIP friendly + legacy SQL-style · parsing de fechas ISO y MM/DD/YYYY |

Las fixtures (`conftest.py`) generan datasets sintéticos de 400 / 2000 filas con el **schema exacto** del maestro BTS+IEM, incluyendo todas las 65 columnas y tipos. Esto permite correr los tests sin depender del CSV real.

---

## Artifacts y serialización

Al finalizar el entrenamiento, `ontimeai.model.save_artifact` deja en `artifacts/` (o el directorio indicado):

| Archivo | Contenido |
|---|---|
| `model.lgb` | Booster de LightGBM (`booster.save_model`) |
| `meta.joblib` | Dict con `threshold`, `feature_cols`, `cat_cols`, `cat_mapping`, `target` |
| `metrics.json` | Métricas de val y test, sizes, confusion matrix, feature_cols |

Carga con `ontimeai.model.load_artifact(path)` → dict con el booster listo para inferencia.

**Reproducibilidad de categóricas**: `cat_mapping` guarda las categorías exactas vistas durante entrenamiento. En inferencia, `features.apply_categorical_mapping(X, cat_mapping)` reaplica el dtype `category` con las mismas categorías; valores desconocidos se convierten en `NaN` (LightGBM los maneja).

---

## CLI de referencia

### `train.py`

```
python3 train.py [opciones]

  --target {binary,multiclass}   Target a entrenar (default: binary)
  --threshold-min FLOAT          Cutoff de minutos para binary (default: 15.0)
  --data PATH                    CSV maestro (default: autodetecta el más amplio)
  --artifacts PATH               Carpeta de salida (default: ./artifacts)
  --num-boost-round INT          Rondas máximas (default: 2000)
  --early-stopping INT           Early stopping patience (default: 100)
  --no-balance                   Desactiva class_weight='balanced'
  --subsample INT                Subsamplea primeras N filas cronológicas (0 = todas)
  --seed INT                     Semilla (default: 42)
  --best-params PATH             Carga lgb_params del JSON de Optuna
```

Emite las métricas en JSON a stdout y guarda los artifacts.

### `tune.py`

```
python3 tune.py [opciones]

  --n-trials INT             Número de trials Optuna (default: 30)
  --data PATH                CSV maestro
  --num-boost-round INT      Rondas por trial (default: 800)
  --subsample INT            Subsample (0 = full)
  --seed INT                 Semilla TPE (default: 42)
  --out PATH                 JSON de salida (default: ./artifacts/best_params.json)
  --no-balance               Desactiva class balancing durante tuning
```

Optimiza val ROC AUC sobre `num_leaves`, `learning_rate`, `min_data_in_leaf`,
`feature_fraction`, `bagging_fraction`, `bagging_freq`, `reg_alpha`, `reg_lambda`.

### `predict.py`

```
python3 predict.py --input CSV [opciones]

  --artifact PATH    Carpeta del artifact (default: ./artifacts)
  --input PATH       CSV con schema del master (requerido)
  --output PATH      Archivo de salida (default: predictions.csv)
```

Si el CSV de input contiene `ARR_DELAY` para vuelos históricos, el script computa
lineage/rolling features en batch-mode. Para inferencia single-flight sin historia
se dejan las features de lineage como NaN (LightGBM las maneja nativamente).

### Descarga de datos

```bash
# BTS: PREZIP mensual, 1987-presente, friendly-schema
python3 descarga_bts.py 2022 2023 2024    # → data_raw/bts/MON_YYYY_raw.csv

# IEM: ASOS METAR, 16 aeropuertos, UTC
python3 descarga_data_iem_multi.py 2022 2023 2024   # → data_raw/iem/clima_iem_asos_YYYY_utc.csv

# Builder multi-año (acepta ambos schemas BTS)
python3 construir_dataset_maestro_multi.py 2022 2023 2024   # → dataset_maestro_ATL_2022-2024_*.csv

# Concat con masters existentes (dedup + orden cronológico)
python3 concatenar_masters.py dataset_maestro_ATL_2022-2024_*.csv dataset_maestro_ATL_2025_*.csv
```

Output:

- **binary**: columnas `proba_delay`, `predicted_delay`
- **multiclass**: columnas `proba_class_0..3`, `predicted_class`

---

## Efecto cascada (TODO post-MVP)

`ontimeai/cascade.py` contiene un stub con el roadmap completo. Está fuera del alcance del MVP actual y se implementará en iteraciones posteriores:

1. **Features de lineage por `TAIL_NUM`**: `prev_arr_delay_tail`, `prev_turnaround_tail`, `tail_flights_today`. **Restricción anti-leakage**: computadas estrictamente con datos disponibles **antes** de `CRS_DEP_TIME` del vuelo objetivo.
2. **Modelo de secuencia** (LSTM / Transformer) sobre rotaciones intra-día por aeronave → propagación intra-aircraft (Qu, Wu & Zhang 2023).
3. **Graph Neural Network** opcional sobre el grafo de red de vuelos → propagación inter-aeropuerto (Zhang et al. 2026 Edge-GNN; Cai et al. 2024 CausalNet; Zeng et al. 2021 Graph-LSTM).
4. **Endpoint API** que devuelve la proyección de cascada downstream para un vuelo de entrada.

El stub levanta `NotImplementedError` explícito si se llama.

---

## Instalación

### Requisitos

- Python 3.10+ (testeado en 3.12)
- Dependencias en `requirements.txt`:

```
lightgbm>=4.0
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
shap>=0.44
joblib>=1.3
pytest>=7.4
optuna>=4.0
requests>=2.28
```

### Setup

```bash
# Clonar
git clone https://github.com/santiago6124/OnTimeAI-Backend.git
cd OnTimeAI-Backend

# (recomendado) entorno aislado
python3 -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# Dependencias
pip install -r requirements.txt

# Verificar
python3 -m pytest
```

### Reconstrucción del dataset maestro (opcional)

Si se necesita reconstruir `dataset_maestro_ATL_2025_BTS_IEM_ORIG_DEST.csv` desde cero:

1. Colocar los CSV mensuales BTS 2025 (`JAN_2025_*.csv`, `FEB_2025_*.csv`, …) en el directorio base.
2. Ejecutar `descarga_data_iem.py` para actualizar `clima_iem_asos_2025_utc.csv`.
3. Ejecutar `construir_dataset_maestro_atl_2025.py`.

Ambos scripts dependen de `BASE_DIR` definido internamente — ajustar a la ruta local.

---

## Referencias académicas

Referencias clave del anteproyecto que informan el diseño de este pipeline:

- **AlBassam & AlShahrani (2025).** *Flight delay prediction: Evaluating machine learning algorithms.* PLoS ONE — comparación de LR / RF / XGB / LGB sobre features similares.
- **Dai (2024).** *Hybrid ML-based model for flight delay.* Scientific Reports — combinación de modelos tabulares con features temporales.
- **Hatipoğlu & Tosun (2024).** *Predictive Modeling of Flight Delays at an Airport.* Applied Sciences — benchmark single-airport.
- **Wandelt, Chen & Sun (2025).** *Flight Delay Prediction: Dissecting Review of Recent Studies.* IEEE TITS — revisión metodológica.
- **Watson et al. (2025).** *Air Travel Delay Prediction: Feature Engineering and ML Approaches.* UC Berkeley — meta-modelos de duración y turnaround.
- **Qu, Wu & Zhang (2023).** *Flight Delay Propagation Prediction Based on Deep Learning.* Mathematics — cascada con LSTM.
- **Cai et al. (2024).** *CausalNet: Spatio-Temporal with Self-Corrective Causal Inference.* arXiv:2407.15185 — cascada con inferencia causal.
- **Zeng et al. (2021).** *Deep Graph-Embedded LSTM for Airport Delay Prediction.* JAT — cascada con grafo.
- **Zhang et al. (2026).** *Edge-Based GNN for Network Delay Prediction.* Aerospace — cascada con GNN.

Ver `AnteProyecto-OnTimeAI.md` para la bibliografía completa.

---

## Licencia y autoría

Proyecto académico — Universidad Católica de Córdoba (UCC). Ver `AnteProyecto-OnTimeAI.md` para los integrantes del equipo y el detalle institucional.

- Frontend: https://github.com/santiago6124/OnTimeAI-Frontend
- Backend: https://github.com/santiago6124/OnTimeAI-Backend
