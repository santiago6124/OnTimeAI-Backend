# TODO — Fase 1 (replay producción sobre BTS 2025)

**Estado**: pendiente. Decidimos saltar a Fase 2 directo (FlightAware AeroAPI live).

## Qué es

Simulación de producción sobre datos cerrados de BTS 2025: recorrer el master cronológicamente día por día, computar lineage/rolling solo con datos anteriores a *t*, predecir, "revelar" el actual al día siguiente, y graficar métricas rolling.

## Por qué la dejamos pendiente

Es un buen *baseline académico* (datos cerrados, reproducible, cero costo) pero no demuestra integración con APIs externas. Fase 2 con AeroAPI sí.

## Cuándo retomarla

- Si Fase 2 queda inestable y necesitamos un fallback para la defensa
- Si querés una métrica histórica complementaria al live test
- Si el jurado pide validación adicional sobre datos del corpus de entrenamiento

## Implementación esperada

Un script `simulate_production.py` que:

1. Carga `dataset_maestro_ATL_2025_BTS_IEM_ORIG_DEST.csv`
2. Itera por día:
   - `t = día en cuestión`
   - `history = df[df.FL_DATE < t]`
   - `today = df[df.FL_DATE == t]`
   - Computa lineage/rolling sobre `history.append(today)`
   - Predice solo los rows de `today`
   - Guarda en SQLite con timestamp de "predicción"
3. Al día siguiente: mark `today.ARR_DELAY` como "actual revelado"
4. Output: serie temporal de accuracy/AUC/F1 a lo largo del año

Estimado: ~150 líneas, ~3 horas de trabajo, 0 USD.
