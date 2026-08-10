# Flight history API

`GET /flight-history/{fa_flight_id}` devuelve la evolución completa de una predicción y la explicación persistida de cada ciclo. La búsqueda conserva continuidad mediante `stable_id` cuando FlightAware cambia el identificador técnico del mismo vuelo.

## Campos por ciclo

- `predicted_at_utc`: instante del ciclo.
- `base_probability`: salida calibrada del modelo antes de ajustes operativos.
- `delay_probability`: probabilidad final publicada.
- `operational_adjustment`: diferencia entre probabilidad final y base.
- `predicted_delay`: clasificación binaria persistida.
- `threshold_used` y `threshold_strategy`: decisión aplicada en ese batch.
- `prediction_phase`: `PRE_DEPARTURE`, `EN_ROUTE` o `POST_LANDING`.
- `operational_context`: demoras GDP de origen/destino, demora de salida intermedia, demora ETA ADS-B y holding disponibles en ese momento.
- `shap`: top-K persistido, ordenado por importancia absoluta, con nombre, etiqueta, magnitud, dirección y valor de la feature.

## Semántica

Los valores SHAP explican el score del booster base. No explican los ajustes externos GDP, salida/ETA u holding. Por este motivo la respuesta separa `base_probability`, `operational_adjustment`, `delay_probability` y `operational_context`.

Las predicciones históricas todavía no guardan `model_version`. Si se necesita reproducibilidad por artefacto, debe agregarse esa columna al esquema de `predictions` y persistirla desde `live_pull.py`; el endpoint no infiere una versión que no quedó registrada.
