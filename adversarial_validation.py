"""
Adversarial Validation: ¿Qué features distinguen train (2022-2024) de live (2025)?

Train un clasificador binario:
  label=0 → vuelo de 2022-2024 (training distribution)
  label=1 → vuelo de 2025 (live proxy)

Si AUC > 0.65: hay drift significativo.
Feature importances = las features que más cambiaron entre train y live.

Nota: el parquet v7 contiene ~53/79 features del modelo. Las rolling rates,
tail lineage, holiday flags y sinusoidal encodings son computadas on-the-fly
en train.py y no están en el parquet base.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, warnings
warnings.filterwarnings("ignore")

PARQUET = "dataset_maestro_FULL_US_2022-2025_v7.parquet"
SAMPLE_EACH = 150_000   # 150k de cada clase
SEED = 42

# Features disponibles en el parquet v7 (excluye columnas leaky y las
# que train.py computa on-the-fly)
LEAKY_COLS = {
    "DEP_DELAY","ARR_DELAY","DEP_DELAY_NEW","ARR_DELAY_NEW","CANCELLED",
    "DIVERTED","ACTUAL_ELAPSED_TIME","AIR_TIME","CARRIER_DELAY",
    "WEATHER_DELAY","NAS_DELAY","SECURITY_DELAY","LATE_AIRCRAFT_DELAY",
    "DEP_TIME","ARR_TIME","DEP_LOCAL_DT","EVENT_ORIGIN_UTC","EVENT_DEST_UTC",
    "ORIG_WX_VALID_UTC","DEST_WX_VALID_UTC","OP_CARRIER_FL_NUM",
    "CRS_DEP_TIME","CRS_ARR_TIME","FL_DATE","YEAR",
}

FEATURE_COLS = [
    "MONTH","DAY_OF_MONTH","DAY_OF_WEEK","OP_CARRIER","TAIL_NUM",
    "ORIGIN","DEST","CRS_ELAPSED_TIME","DISTANCE","FLOW_ATL","PAR_AIRPORT",
    "CRS_DEP_MIN","ORIG_WX_TMPC","ORIG_WX_DWPC","ORIG_WX_RELH","ORIG_WX_DRCT",
    "ORIG_WX_SKNT","ORIG_WX_ALTI","ORIG_WX_P01M","ORIG_WX_VSBY","ORIG_WX_GUST",
    "ORIG_WX_CODES","ORIG_WX_PRECIP_FLAG","ORIG_WX_LOW_VIS_FLAG",
    "ORIG_WX_STRONG_WIND_FLAG","ORIG_WX_MATCH_GAP_MIN","DEST_WX_TMPC",
    "DEST_WX_DWPC","DEST_WX_RELH","DEST_WX_DRCT","DEST_WX_SKNT","DEST_WX_ALTI",
    "DEST_WX_P01M","DEST_WX_VSBY","DEST_WX_GUST","DEST_WX_CODES",
    "DEST_WX_PRECIP_FLAG","DEST_WX_LOW_VIS_FLAG","DEST_WX_STRONG_WIND_FLAG",
    "DEST_WX_MATCH_GAP_MIN","BEARING_DEG","ERA5_U_KT","ERA5_V_KT",
    "ERA5_HEADWIND_KT","ERA5_CROSSWIND_KT","ERA5_TAILWIND_FLAG",
    "PREV_ACTUAL_BLOCK_MIN","PREV_SCHED_BLOCK_MIN","PREV_BLOCK_DELTA_MIN",
    "PREV_HOLDING_MIN","PREV_ROUTE_DEVIATION_PCT","PREV_ADSB_AVAILABLE",
]

cat_cols = ["OP_CARRIER","TAIL_NUM","ORIGIN","DEST","FLOW_ATL","PAR_AIRPORT","ORIG_WX_CODES","DEST_WX_CODES"]

print("Cargando parquet (solo columnas necesarias)...")
df = pd.read_parquet(PARQUET, columns=["YEAR"] + FEATURE_COLS)

print(f"  Total: {len(df):,} filas, {df.shape[1]} cols")

# Separar train (2022-2024) y live proxy (2025)
train_pool = df[df["YEAR"] < 2025].drop(columns=["YEAR"])
live_pool  = df[df["YEAR"] == 2025].drop(columns=["YEAR"])
print(f"  Train pool: {len(train_pool):,}  |  Live pool: {len(live_pool):,}")

# Samplear balanceado
rng = np.random.default_rng(SEED)
train_sample = train_pool.sample(n=SAMPLE_EACH, random_state=SEED)
live_sample  = live_pool.sample(n=SAMPLE_EACH, random_state=SEED)

train_sample["_adv_label"] = 0
live_sample["_adv_label"]  = 1

combined = pd.concat([train_sample, live_sample], ignore_index=True)
combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

y = combined["_adv_label"].values
feature_cols_available = [c for c in FEATURE_COLS if c in combined.columns]
X = combined[feature_cols_available].copy()

# Detectar categoricas
cat_actual = [c for c in cat_cols if c in X.columns]
for c in cat_actual:
    X[c] = X[c].astype("category")

# Fix bool columns stored as object
bool_cols = [c for c in X.columns if "FLAG" in c and c not in cat_actual]
for c in bool_cols:
    X[c] = X[c].astype("float32")

# Notar features NO disponibles en el parquet (rolling rates, etc.)
not_in_parquet = [
    "prev_arr_delay_tail","prev_turnaround_tail_min","tail_flights_today_prior",
    "carrier_delay_rate_yday","origin_delay_rate_yday","carrier_delay_rate_24h",
    "carrier_delay_rate_7d","origin_delay_rate_1h","origin_delay_rate_6h",
    "origin_delay_rate_24h","dest_delay_rate_1h","dest_delay_rate_6h",
    "dest_delay_rate_24h","absorb_score_origin","is_us_holiday",
    "days_to_nearest_holiday","is_thanksgiving_window","is_summer_peak",
    "dep_hour_sin","dep_hour_cos","dep_dow_sin","dep_dow_cos",
    "dep_month_sin","dep_month_cos","congestion_orig_window",
    "congestion_dest_window","wx_both_precip","wx_both_low_vis","wx_both_strong_wind",
]
print(f"\nFeatures en parquet analizadas: {len(feature_cols_available)}")
print(f"Features computadas on-the-fly (no en parquet): {len(not_in_parquet)}")
print(f"  → {not_in_parquet[:5]}... (requieren análisis separado)")

# 5-fold CV adversarial
print("\nEntrenando clasificador adversarial (5-fold CV)...")
params = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "verbose": -1,
    "random_state": SEED,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
aucs = []
importances = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        categorical_feature=cat_actual,
    )

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    aucs.append(auc)
    importances.append(dict(zip(feature_cols_available, model.feature_importances_)))
    print(f"  Fold {fold+1}: AUC = {auc:.4f}")

mean_auc = np.mean(aucs)
print(f"\n{'='*50}")
print(f"Adversarial AUC promedio: {mean_auc:.4f}")
if mean_auc < 0.60:
    print("  → Distribuciones muy similares. El gap NO es por drift de features.")
elif mean_auc < 0.70:
    print("  → Drift MODERADO. Algunos features cambiaron entre 2022-24 y 2025.")
else:
    print("  → Drift ALTO. Distribución muy diferente entre train y 2025.")

# Promediar importances y mostrar top 25
avg_imp = pd.Series({
    feat: np.mean([imp[feat] for imp in importances])
    for feat in feature_cols_available
}).sort_values(ascending=False)

print(f"\n{'='*50}")
print("TOP 25 FEATURES que más distinguen train vs 2025 (más alto = más drift):")
print(f"{'Feature':<35} {'Importance':>12}")
print("-" * 50)
for feat, imp in avg_imp.head(25).items():
    print(f"  {feat:<33} {imp:>10.1f}")

# Guardar resultados
results = {
    "adversarial_auc_mean": float(mean_auc),
    "adversarial_auc_per_fold": [float(a) for a in aucs],
    "n_sample_each_class": SAMPLE_EACH,
    "train_years": "2022-2024",
    "live_proxy_year": "2025",
    "top_drift_features": avg_imp.head(30).to_dict(),
}
with open("artifacts/adversarial_validation_v7.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResultados guardados en artifacts/adversarial_validation_v7.json")
