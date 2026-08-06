# Optimizaciones y Calibración del Backend (Agosto 2026)

Este documento resume los cambios, optimizaciones de infraestructura e integración de modelos realizados en el backend UCC Grupo 9.

---

## 1. Integración de Timezone (Sliding Window)
* **Archivo:** [api.py](file:///Users/mateopappalardo/FACU/TEsis/OnTimeAI-Backend/api.py)
* **Cambio:** Reemplazo de la función `_latest_predictions_today` por `_latest_predictions_active`.
* **Beneficio:** En lugar de realizar consultas estrictas del calendario UTC de hoy (lo que causaba un "cliff" de vuelos vacíos a la medianoche), se consulta una ventana de tiempo deslizante de **-6 horas a +18 horas** respecto a la hora UTC actual. También mantiene en el dashboard aquellos vuelos retrasados de las últimas 24 horas que aún no hayan despegado.

---

## 2. Depuración de Base de Datos y Prevención de Memoria (OOM)
* **Script:** [scripts/prune_db.py](file:///Users/mateopappalardo/FACU/TEsis/OnTimeAI-Backend/scripts/prune_db.py)
* **Integración:** Se añadió la llamada automática en `live_job.py` antes de subir la base de datos de vuelta a GCS.
* **Beneficio:**
  - Depura de forma segura registros de más de 30 días en las tablas `predictions`, `prediction_shap`, `flights`, `weather_obs`, `nas_status` y `runs`, y ejecuta un `VACUUM` de SQLite para compactar el archivo en disco.
  - Mitiga el riesgo crítico de **Out of Memory (OOM)** en Cloud Run (el límite del contenedor es 2 GB, y la base de datos había crecido a **1.61 GiB**).
  - Redujo el peso de la base de datos productiva de **1,732 MB** a solo **68.93 MB** (un **96%** menos), acelerando drásticamente el ancho de banda y latencia de red.
* **Backup Histórico:** Se mantiene el backup completo descargado en la máquina local en `OnTimeAI-Backend/live_data_prod_backup.db` para poder realizar re-entrenamientos fuera de línea con toda la historia del proyecto.

---

## 3. Recalibración del Modelo (`4year_v9_recal`)
* **Carpeta:** [artifacts/4year_v9_recal](file:///Users/mateopappalardo/FACU/TEsis/OnTimeAI-Backend/artifacts/4year_v9_recal)
* **Cambio:** Recalibración Platt Scaling (Sigmoide) entrenada sobre **53,750 muestras** históricas acumuladas en el backup de producción.
* **Resultados:**
  - **AUC:** Mantiene el poder discriminativo original en **`0.9063`**.
  - **Brier Score (Error):** Disminuyó de `0.1108` a **`0.0779`** (mejora del **30%**).
  - **ECE (Error de Calibración):** Se desplomó de `12.59%` a solo **`1.41%`** (mejora del **88%**). Las probabilidades predichas por el modelo ahora corresponden exactamente con la tasa real de demoras en Atlanta.
  - **Nuevo Threshold de Alerta:** Fijado en **`0.3440`** (target pos rate = 22%).

---

## 4. Endpoint Administrativo `/admin/db-stats`
* **Archivo:** [api.py](file:///Users/mateopappalardo/FACU/TEsis/OnTimeAI-Backend/api.py)
* **Beneficio:** Expone métricas clave del tamaño de la base de datos en MB, número de registros por tabla y rango de fechas de predicciones almacenadas. Requiere autenticación de `superadmin`.
