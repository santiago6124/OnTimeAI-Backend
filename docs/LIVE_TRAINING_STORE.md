# OnTimeAI — dataset live apto para entrenamiento

> Estado al 2026-08-11: implementación publicada en `main` y activada como
> canary en `ontimeai-live-pull-2`. El primer ciclo natural completó correctamente
> y publicó 5.251 eventos en 34 Parquet inmutables más su contrato. El reloj de
> validación empezó a las 14:23 UTC; el predictor principal sigue pendiente del
> canary de 48–72 horas.

## Decisión

`live_data.db` sigue siendo solamente el estado operativo de API/UI. No debe
usarse como histórico de entrenamiento porque es mutable, tiene más de un
escritor, se poda y no conserva el vector exacto que recibió LightGBM.

El histórico ML se guarda como un ledger append-only de Parquet en un bucket GCS
separado. Cada snapshot representa una predicción en su instante exacto de
conocimiento; el resultado real se escribe posteriormente como una revisión
separada. El label se deriva recién al materializar un dataset.

```text
live_pull
  ├─ predictions + actuals en live_data.db
  └─ outbox transaccional (mismo commit que la predicción)
       └─ live_job publica sólo desde la generación SQLite ganadora
            └─ GCS raw Parquet inmutable
                 └─ materializador por horizonte
                      └─ silver Parquet + manifest reproducible
```

No se usa otro SQLite histórico: volvería a introducir un objeto mutable grande
y la misma condición de carrera que se busca evitar.

## Garantía de entrega

Las tablas `training_export_outbox` y `training_export_delivery` se crean de
forma compatible dentro de `live_data.db`:

1. `live_pull` inserta la predicción y su snapshot causal en la misma transacción.
2. `live_job` sube la DB usando el precondition de generación existente.
3. Sólo después de ganar esa carrera publica objetos Parquet con
   `if_generation_match=0`.
4. La siguiente ejecución reconoce el mismo nombre determinístico, registra la
   entrega y elimina el outbox antes de su único upload de la DB.

Si una ejecución pierde la carrera de GCS, sus snapshots nunca se publican. Si
el Parquet se publicó pero se perdió el acknowledge, el retry obtiene el mismo
objeto y no lo duplica. Si el training bucket está temporalmente caído, el outbox
permanece dentro de la DB ganadora.

Cada enqueue usa un `SAVEPOINT`: un error en cualquier fila revierte todo ese
lote sin revertir la transacción de serving anterior. El bootstrap de outcomes
se limita a 5.000 filas por ciclo y prioriza fuentes nunca vistas/corregidas;
el publisher procesa como máximo ocho grupos de `created_at_utc` por llamada.
Esos límites evitan cargar un backlog completo en RAM y siguen drenándolo sin
saltos ni starvation.

Para outcomes, el lookback de siete días es el camino rápido, no el límite de
recuperación: un watermark por `fa_flight_id` vuelve a seleccionar cualquier
estado nunca visto o con settlement posterior aunque la caída haya durado más.
Sí existe un límite inherente a `actuals`, que es mutable: una transición
intermedia A→B→A completamente sobrescrita durante una caída no puede
reconstruirse, aunque el estado final A sí se recupera. Capturar cada transición
intermedia exigiría integrar el outbox dentro de cada writer de actuals.

## Contrato `live-training-v1`

Cada evento de predicción incluye:

- IDs originales de fuente, `canonical_flight_key` y `rotation_key` versionados.
- carrier, número, tail, ruta, fecha local y horarios scheduled/estimated tal
  como se conocían al predecir.
- `predicted_at_utc`, fase y anticipación respecto de la salida programada.
- nombre y SHA-256 del booster/meta, hash del schema de features, hash del
  mapping categórico, SHA-256 determinístico del código transformador y commit
  de origen cuando el deploy configura `SOURCE_COMMIT`.
- salida cruda del booster, probabilidad calibrada y probabilidad final después
  de ajustes operativos, por separado.
- threshold/estrategia, GDP, estimated delay, intermediate delay, ADS-B,
  holding, congestión ATL y carrier smoothing.
- las 84 features `raw__*` antes del cold-deck y las 84 `model__*` exactas que
  recibió LightGBM después del fallback.
- cantidad/tasa de faltantes antes y después de imputación.

Los categóricos se guardan como strings canónicos y los faltantes como null real
de Parquet, nunca como códigos internos ni como el string `"NaN"`.

Cada outcome conserva horarios reales, delays continuos, cancelación/desvío,
proveedor (`fr24`, `aeroapi` o desconocido para filas históricas), revisión
monótona y timestamp de settlement. No se destruye el delay continuo aunque hoy
el target sea `arr_delay_min > 15`.

Si dos source records describen la misma rotación física pero discrepan en el
estado o el delay, el materializador no elige el observado más tarde: cuarentena
todo el vuelo. La copia productiva auditada contiene casos reales que cruzan el
umbral de 15 minutos, por lo que elegir por frescura introduciría labels erróneos.

## Estructura del bucket

```text
live-training/
  contracts/
    schema=live-training-v1/*.json  # orden, categóricos y vocabulario por hash
  raw/
    prediction_snapshots/schema=live-training-v1/contract=*/event_date=YYYY-MM-DD/*.parquet
    flight_outcomes/schema=live-training-v1/contract=*/flight_date=YYYY-MM-DD/observed_date=YYYY-MM-DD/*.parquet
  silver/                       # futuros datasets materializados
  manifests/                    # manifests inmutables de trainsets
  quarantine/                   # shards que fallen calidad
  tmp/                          # objetos descartables
```

Los raw deben conservarse al menos cinco años. `tmp/` y `quarantine/` pueden
eliminarse a los 30 días. No bloquear la retention policy hasta validar varias
semanas de operación; bloquearla es irreversible.

## Configuración de Cloud Run Job

La captura sólo se activa si existe `TRAINING_DATA_BUCKET` o si se fuerza
`TRAINING_STORE_ENABLED=true`. Un override activo sin bucket es configuración
inválida: no se permite acumular un outbox que nunca pueda entregarse.
También se rechaza `TRAINING_DATA_BUCKET` cuando `GCS_BUCKET` está vacío: el
modo local de `live_job` recrea `/tmp/live_data.db` desde la copia bundleada y no
puede prometer persistencia tras un reinicio. Para desarrollo, ejecutar
`live_pull.py` contra una DB local persistente y publicar sólo desde una copia de
prueba explícita.

| Variable | Default | Uso |
|---|---:|---|
| `TRAINING_DATA_BUCKET` | vacío | bucket GCS dedicado; vacío desactiva captura |
| `TRAINING_DATA_PREFIX` | `live-training` | prefijo append-only |
| `TRAINING_STORE_ENABLED` | auto | override explícito |
| `TRAINING_STORE_REQUIRED` | `false` | hace fallar el job si no puede entregar |
| `TRAINING_OUTCOME_LOOKBACK_DAYS` | `7` | fast path; los nunca vistos se recuperan fuera de la ventana |
| `TRAINING_OUTCOME_ENQUEUE_MAX_EVENTS` | `5000` | máximo de outcomes que entran al outbox por ciclo |
| `TRAINING_PUBLISH_MAX_GROUPS` | `8` | máximo de lotes temporales publicados por llamada |
| `TRAINING_OUTBOX_WARN_EVENTS` | `20000` | umbral de alerta del backlog durable |
| `PRUNE_VACUUM_MIN_DELETED` | `50000` | `VACUUM` sólo si el ciclo borró más filas que este umbral |
| `SOURCE_COMMIT` | vacío | commit/revisión del transformador |

Mantener `TRAINING_STORE_REQUIRED=false` durante el canary y el inicio
productivo, acompañado por alertas de backlog y errores de publicación. Con
`true`, una indisponibilidad del bucket de training también hace fallar el job
de predicciones; sólo debe activarse si se acepta explícitamente ese
acoplamiento de disponibilidad. La durabilidad no depende de `true`: en modo
opcional el outbox permanece en la SQLite ganadora para reintentar.

El bucket creado para el proyecto actual es:

```text
gs://ontimeai-150917658060-training-us-central1
```

Está configurado como regional `us-central1`, con Uniform Bucket-Level Access,
Public Access Prevention y soft delete de 7 días. También se crearon cuentas
dedicadas de exporter y lectura con permisos sobre este bucket.

El canary conserva temporalmente la cuenta default de Compute porque la cuenta
exporter todavía no pudo recibir el binding administrativo para leer el secreto
operativo existente. Esa cuenta default sigue teniendo rol `Editor` y la
comparten servicios, jobs y builds; es deuda de mínimo privilegio. No migrar el
runtime a la SA dedicada hasta completar ese acceso. El exporter necesita crear
y leer objetos —la lectura verifica colisiones idempotentes—, pero no borrarlos;
la SA del materializador debe seguir siendo sólo lectora.

## Materialización causal

El comando disponible es:

```bash
python scripts/materialize_live_training.py \
  --bucket ontimeai-150917658060-training-us-central1 \
  --prefix live-training \
  --start-date 2026-08-11 \
  --end-date 2026-11-30 \
  --horizons-hours 6 2 \
  --max-snapshot-age-minutes 60 \
  --output /tmp/ontimeai-live-silver.parquet
```

Para un horizonte `H` se define:

```text
cutoff = scheduled_out_utc - H
snapshot = última predicción con predicted_at_utc <= cutoff
```

El snapshot debe estar como máximo 60 minutos antes del cutoff y también debe
ser anterior a `actual_off_utc`. No se selecciona por score, por hora real de
salida ni por cercanía posterior al cutoff. Sólo se aceptan `PRE_DEPARTURE`,
outcomes finales, no cancelados/no desviados y con delay válido.

El resultado contiene una fila por vuelo físico/horizonte, las features exactas
con sus nombres originales, `TARGET` y metadata de auditoría. Junto al Parquet se
genera un manifest JSON con objetos/generaciones, hashes, parámetros, conteos y
SHA-256 del output.

`--start-date/--end-date` delimitan la fecha local del vuelo. El loader amplía
automáticamente los shards de borde para no perder un snapshot T-6/T-12 que haya
caído en el día UTC anterior, y recorta las filas recién después de resolver el
horizonte.

El manifest también contiene el contrato categórico. El loader ofrece dos modos:

- `model_exact`: reproduce el vector y mapping del champion para replay,
  evaluación y recalibración.
- `raw_challenger`: conserva los strings previos al mapping, incluidas categorías
  nuevas. Su vocabulario debe ajustarse usando solamente el split de train.

No se deben mezclar horizontes en un mismo entrenamiento: el adapter exige uno o
una selección explícita. Tampoco permite mezclar `model_version`, hashes de
booster/meta o `transformer_version`, aun cuando el schema de features coincida;
cada rollout se materializa o filtra por separado. El split temporal mantiene
todas las filas de un vuelo físico en una sola partición.

Los outcomes se particionan por una fecha física anclada en la primera revisión
de cada `source_record_id` y secundariamente por fecha de observación. El ancla
es inmutable aunque una corrección posterior cambie el schedule; así todas las
revisiones se cargan juntas y nunca resucita un label viejo. El manifest registra
el SHA-256 del código materializador y las versiones exactas de Python, pandas,
PyArrow y NumPy, además del hash del output y de cada objeto fuente.

## Cuánto tiempo debe recolectar

La respuesta depende del uso y del horizonte:

| Uso | Mínimo exploratorio | Recomendado |
|---|---:|---:|
| Validar integridad/paridad | 48–72 h | 7 días |
| Recalibrar T-2h | 1.000 vuelos y 100 positivos | 2.000+ vuelos, 300+ positivos y 4 semanas |
| Recalibrar T-6h | no estimable todavía | iniciar el reloj recién con cobertura ≥90% |
| Challenger híbrido BTS + live | 10.000 live clean | 8–12 semanas de contexto live + corpus histórico |
| Challenger sólo live | 20.000 clean | 3–4 meses por volumen; 7–8 meses para 5.000 positivos al ritmo observado |
| Reemplazar modelo multianual | no usar live solo | 6–12 meses + corpus BTS histórico |

La estimación usa una auditoría read-only de producción entre el 7 y el 10 de
agosto de 2026: 783 vuelos tenían predicción PRE elegible en la ventana exacta
T-2h ±60 min y resultado limpio; 88 fueron positivos. Eso equivale a unos 196
labels/día y 22 positivos/día. Es sólo un proxy de capacidad: esas predicciones
viejas no guardaron el vector exacto y no forman parte del nuevo dataset.

Para T-6h no apareció una muestra etiquetada consistente en esa ventana exacta.
Para T-12h tampoco existe cobertura: el predictor por default mira seis horas.
Antes de iniciar esos relojes se deben corregir/medir T-6 y coordinar
`PREDICT_HORIZON_HOURS=12` con `CAPTURE_FUTURE_LEGS=true` para T-12.

La promoción de un challenger exige además:

- labels maduros ≥95%; cobertura del horizonte ≥90%;
- schema único y vector reproducible ≥95%;
- holdout temporal posterior de al menos cuatro semanas;
- AUC, PR-AUC, Brier, log-loss y ECE con bootstrap;
- métricas por ARR/DEP, carrier, ruta, hora y estación;
- una o dos semanas de shadow frente al champion productivo.

## Estado del rollout y operación siguiente

1. Completado: bucket regional y cuentas dedicadas creados con UBLA, PAP y soft
   delete de 7 días.
2. Completado: digest nuevo desplegado primero en `ontimeai-live-pull-2`, con
   `TRAINING_STORE_REQUIRED=false`; primer ciclo natural exitoso en 7m45s.
3. Completado en el primer ciclo: 5.000 revisiones de outcome y 251 snapshots,
   sin IDs duplicados, hashes de payload verificados y contrato `4year_v9` con
   84 features raw/model. La SQLite ganadora pasó `quick_check` e
   `integrity_check`.
4. En curso: observar 48–72 h, medir duración, memoria y drenaje del backlog, y
   ejecutar un round-trip de control a silver cuando maduren los primeros labels.
5. Pendiente: activar el mismo digest en el predictor principal manteniendo
   inicialmente `TRAINING_STORE_REQUIRED=false`; cambiarlo sólo si se acepta el
   tradeoff de disponibilidad.
6. Pendiente recurrente: materializado semanal de control y scorecard diario.
7. Recién a las cuatro semanas considerar recalibración; a las 8–12 semanas,
   entrenar un challenger combinado con BTS.

El materializador inicial concatena los shards seleccionados en memoria. Para
ventanas de seis a doce meses se debe materializar por particiones mensuales o
migrar el scan a `pyarrow.dataset`/DuckDB antes de ejecutar una corrida completa.
