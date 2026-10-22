# Gate de complejidad ciclomática

Desde el 2026-10-22 el CI rechaza funciones nuevas con complejidad ciclomática
mayor a **10** en el código que se despliega. Este documento explica qué mide,
por qué ese umbral, qué quedó exento y cómo se achica la deuda.

## Qué mide y con qué

La complejidad ciclomática (McCabe, 1976) cuenta los caminos independientes de
una función: 1 más una unidad por cada decisión (`if`, `elif`, `for`, `while`,
`except`, `with`, comprensiones, `and`/`or` en condiciones). A más caminos, más
casos de prueba hacen falta para cubrirla y más fácil es que un cambio rompa un
camino que nadie ejercita.

Se usan dos herramientas con dos roles distintos, las dos fijadas en
`requirements.txt` para que local y CI cuenten igual:

| Herramienta | Rol | Qué cuenta |
|---|---|---|
| `ruff` regla C901 | **gate** (bloquea) | McCabe estándar por función |
| `radon cc` | **reporte** (informa) | McCabe más operadores booleanos; incluye clases y métodos |

Radon da valores más altos que ruff para la misma función (`live_pull.py:main`
es 72 para ruff y 157 para radon). El gate usa ruff porque es el conteo
estándar; radon queda para seguir la evolución y para el informe.

## Alcance

Sólo el código productivo, definido en `PRODUCTIVO` de
`scripts/complejidad.py`: `api.py`, `live_pull.py`, `live_job.py`,
`watchdog.py`, `predict.py`, el paquete `ontimeai/` y los dos scripts que corren
en producción (`materialize_live_training.py`, `prune_db.py`). Los scripts de
análisis, `feature_engineering_v7/` y `unbiased_sample_live/` quedan fuera: no
se despliegan y reescribirlos no reduce ningún riesgo operativo.

## Umbral y baseline

Umbral 10, el clásico: hasta 10 la función es simple; de 11 a 20 moderada; más
de 20 compleja y difícil de probar exhaustivamente.

Al instalar el gate había **21 funciones productivas por encima de 10**. Están
en `.complejidad-baseline.json`, con el valor con el que entraron. La baseline
es un trinquete:

- una función nueva por encima de 10 **falla** el CI, aunque lleve `# noqa`;
  el check corre con `--ignore-noqa` justamente para eso;
- una función exenta que **crece** por encima de su valor de entrada falla;
- una función exenta que baja de 10 se saca con `--actualizar-baseline`, que
  sólo acepta listas más chicas.

Las funciones exentas llevan `# noqa: C901` en el `def` para que `ruff check`
local pase limpio; la marca no exime nada por sí sola.

## Cómo se usa

```bash
python scripts/complejidad.py --check                  # lo mismo que corre CI
python scripts/complejidad.py --reporte complejidad.json --resumen resumen.md
python scripts/complejidad.py --actualizar-baseline    # tras bajar una función
```

Para bajar una función de la lista: extraer bloques con nombre propio (cada
fase del ciclo, cada validación), reemplazar cascadas de `if` por tablas o
despacho, y cubrir con tests **antes** de mover código. El orden importa: la
función más compleja del sistema (`live_pull.py:main`) está en el archivo con
menos cobertura (6 %), así que primero se le escriben tests de caracterización
y recién después se parte.

## Deuda inicial (2026-10-22)

| Función | CC (ruff) | Archivo |
|---|---|---|
| `main` | 72 | `live_pull.py` |
| `materialize_fixed_horizons` | 41 | `ontimeai/training_store.py` |
| `_parse_nas_xml` | 27 | `ontimeai/faa_nas.py` |
| `build_inference_frame` | 24 | `ontimeai/live.py` |
| `apply_lineage_fallback` | 23 | `ontimeai/lineage_fallback.py` |
| `_add_v7_wind_features` | 23 | `ontimeai/live.py` |
| `prepare_silver_training_data` | 22 | `ontimeai/training_store.py` |
| `publish_pending_outbox` | 17 | `ontimeai/training_store.py` |
| `load_master` | 16 | `ontimeai/data.py` |
| `_coerce_parquet_frame` | 16 | `ontimeai/training_store.py` |
| `main` | 13 | `live_job.py` |
| `fetch_iem_obs`, `compute_adsb_holding_min` | 12 | `ontimeai/live.py` |
| `prepare_dataset` | 12 | `ontimeai/pipeline.py` |
| `chain_walk_inbound`, `compute_adsb_eta_delay` | 11 | `ontimeai/live.py` |
| `enqueue_prediction_snapshots`, `_upload_create_only` | 11 | `ontimeai/training_store.py` |
| `prepare_inference_frame` | 11 | `predict.py` |
| `_feature_contract_from_bytes`, `main` | 11 | `scripts/materialize_live_training.py` |

Promedio radon del código productivo al instalar el gate: 6,4 (rango B).
