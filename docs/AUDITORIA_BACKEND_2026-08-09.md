# Auditoría técnica del Backend — 2026-08-09

## Resumen ejecutivo

El backend compila y su suite principal aprueba, pero la auditoría encontró
riesgos importantes de seguridad, persistencia, concurrencia y paridad entre
entrenamiento e inferencia. El hallazgo más urgente de consistencia era el
patrón compartido `download -> modify -> upload` sobre
`gs://ontimeai-live-db/live_data.db`: un escritor lento podía sobrescribir la
salida de otro.

La corrección de concurrencia incluida junto con este documento reemplaza el
modelo last-write-wins por control optimista de generación en los tres
escritores actuales: `ontimeai-live-pull`, `ontimeai-live-pull-2` y
`ontimeai-harvester`. Los horarios escalonados se mantienen para reducir
conflictos, pero la integridad ya no depende exclusivamente de la duración de
los jobs.

Este documento no contiene valores de credenciales, secretos, tokens ni hashes.

## Addendum operativo — 2026-08-11

- La corrección CAS de F-02 ya está desplegada en el harvester y en el canary
  `ontimeai-live-pull-2`; el predictor principal permanece pendiente del canary
  de 48–72 horas. Se mantiene el stagger `:00/:08/:15/:30/:38/:45` UTC.
- El harvester nuevo completó su primer ciclo natural y subió la generación
  ganadora. El primer ciclo del canary también completó correctamente, desde la
  generación descargada hasta una nueva generación, sin conflicto ni error.
- El fix de pruning de F-04 está incluido en el canary: `scripts/prune_db.py`
  forma parte de la imagen y `VACUUM` sólo se ejecuta al superar el umbral de
  filas borradas. El primer ciclo ejecutó ambos caminos y terminó dentro del
  timeout.
- El training store append-only publicó 5.251 eventos en 34 Parquet y un
  contrato. Los payloads, la SQLite ganadora y sus invariantes se verificaron;
  esto no cierra la trazabilidad pendiente de la tabla serving `predictions`.

## Alcance y verificaciones

- Código revisado: API FastAPI, autenticación, SQLite/GCS, Cloud Run Job,
  pipeline live, entrenamiento, feature engineering, evaluación y scripts.
- Backend: `116 passed`, `5 skipped`, `22 warnings` luego de inicializar el
  lifecycle de usuarios que la suite actual omite.
- Scrapper: `13 passed`, incluyendo cuatro pruebas nuevas de generación GCS y
  reintento integral del harvester.
- Cobertura medida del núcleo: aproximadamente `47%`; `live_pull.py` y
  `live_job.py` no estaban importados por la suite anterior a esta corrección.
- Compilación Python: aprobada mediante `compileall`.
- Chequeo estático Ruff (`E9`, `F`, `B`): 179 observaciones, mayormente higiene;
  se confirmó al menos un error de nombre indefinido con impacto en runtime.
- Base productiva observada durante la auditoría: `PRAGMA quick_check` e
  `integrity_check` aprobados.

## Hallazgos priorizados

### F-01 — Credenciales y JWT con fallbacks predecibles

**Severidad:** crítica
**Estado:** abierto

`api.py` crea usuarios iniciales y firma tokens con valores fallback cuando no
existen variables de entorno. La configuración productiva verificada durante
la auditoría coincidía con credenciales fallback conocidas por el código. No se
registran aquí sus valores.

Riesgos adicionales:

- El endpoint público de login no tiene rate limit ni límites estrictos de
  tamaño para username/password.
- Desactivar o cambiar el rol de un usuario no revoca inmediatamente un JWT ya
  emitido; permanece válido hasta ocho horas.
- Secretos sensibles están configurados como variables directas de Cloud Run en
  vez de referencias a Secret Manager.

**Recomendación:** rotar credenciales y JWT, eliminar fallbacks, hacer fail-fast
en startup si faltan secretos, usar Secret Manager, agregar rate limiting y
validar estado/rol del usuario en cada sesión o incorporar revocación.

### F-02 — Actualización perdida por escritores concurrentes de la DB compartida

**Severidad:** crítica
**Estado:** desplegado en harvester y canary; predictor principal pendiente

Antes de esta corrección, Backend y Scrapper descargaban la misma SQLite,
realizaban cambios independientes y subían sin precondición. El último upload
ganaba incluso si partía de una generación obsoleta.

Corrección aplicada:

1. El escritor ejecuta `blob.reload()` y fija la generación `N`.
2. Descarga exactamente `N` usando `if_generation_match=N`.
3. Modifica y valida la copia local.
4. Sube con `if_generation_match=N`.
5. Si otro job publicó primero, GCS responde `PreconditionFailed`.
6. El perdedor descarta su snapshot, descarga la generación ganadora y repite
   el pipeline completo.
7. Después de `GCS_GENERATION_RETRIES` conflictos —default `2`, tres intentos
   totales— el job termina con código `3` y nunca pisa una versión nueva.

El caso de creación inicial usa `if_generation_match=0`, por lo que tampoco
puede sobrescribir un objeto creado simultáneamente.

La corrección está implementada en:

- Backend: `live_job.py` para ambos jobs de predicción.
- Scrapper: `ontimeai_scrapper/db.py` y
  `ontimeai_scrapper/harvester.py`.

Los schedulers UTC deben conservarse escalonados en `:00/:08/:15/:30/:38/:45`.
El stagger reduce costo y reintentos; la precondición de generación garantiza
que un solapamiento excepcional no produzca pérdida silenciosa.

### F-03 — Refresh no atómico de SQLite dentro de la API

**Severidad:** alta
**Estado:** abierto

La API descarga GCS directamente sobre `/tmp/live_data.db` mientras pueden
existir conexiones SQLite abiertas. Tampoco existe un lock que impida que dos
requests refresquen simultáneamente. Esto puede exponer un archivo parcial o
reemplazarlo durante una consulta.

El intervalo fijo de refresh es de 1000 segundos, por lo que el dashboard puede
quedar uno o dos ciclos por detrás.

**Recomendación:** descargar a un nombre temporal por generación, ejecutar
`quick_check`, hacer `os.replace()` atómico bajo lock y consultar la generación
de GCS en vez de depender solamente de un temporizador.

### F-04 — Estado exitoso aunque el upload no se haya realizado y pruning ausente

**Severidad:** alta
**Estado:** corregido en canary; predictor principal pendiente

El job original abortaba `_gcs_upload()` ante corrupción pero después retornaba
el código exitoso de `live_pull`. La corrección de F-02 ahora propaga el fallo de
integridad y evita un falso verde.

El canary ya incluye `scripts/prune_db.py` en la imagen. El pruning cuenta las
filas realmente borradas y sólo ejecuta `VACUUM` cuando supera
`PRUNE_VACUUM_MIN_DELETED`; el primer ciclo productivo verificó ejecución y
skip sin exceder el timeout. Falta promover el mismo digest al predictor
principal después de la ventana de observación.

### F-05 — Diferencia de semántica en `TAIL_DELAY_DECAY`

**Severidad:** alta
**Estado:** abierto

En entrenamiento, `TAIL_DELAY_DECAY` se calcula sin aplicar el cold-deck
fallback. En inferencia, primero se imputan las features de lineage y después se
calcula el decay. Por lo tanto, una fila que fue missing durante entrenamiento
puede convertirse en un valor sintético durante serving.

La variable concentra aproximadamente 46.8% de la ganancia total reportada por
el booster v9, por lo que este train/serve skew puede alterar de forma material
la calibración.

**Recomendación:** usar un único transformador versionado para entrenamiento e
inferencia, aplicar exactamente la misma política de missing/fallback y agregar
un test de paridad por fila.

### F-06 — Weather look-ahead y features ERA5 ausentes en producción

**Severidad:** alta
**Estado:** abierto

La construcción histórica une el METAR más cercano con
`direction="nearest"`, permitiendo observaciones hasta 90 minutos posteriores
al evento. Para destino también usa clima alrededor de la llegada programada,
información que no existe al emitir una predicción pre-salida. Esto puede inflar
el backtest y produce una distribución diferente a producción.

El modelo v9 incluye `BEARING_DEG` y cinco features de viento ERA5, pero la
imagen live no contiene los grids de `data_raw/era5_wind` y los requirements no
incluyen `netCDF4`. El bloque degrada silenciosamente a `NaN`. Esas variables
representan cerca de 0.5% de la ganancia del modelo.

Las seis features `PREV_ADSB_*` tampoco se construyen en live, aunque el booster
actual les asigna ganancia prácticamente nula.

**Recomendación:** construir features por `prediction_timestamp`, usar sólo
observaciones o forecasts emitidos antes de ese timestamp y eliminar o servir
realmente las columnas incluidas en el contrato del modelo.

### F-07 — Modelo y threshold reportados no trazan la predicción real

**Severidad:** alta
**Estado:** abierto

`live_pull.py` usa `artifacts/4year_v9` como default independiente de
`ACTIVE_MODEL`, mientras la API puede informar `4year_v9_recal`. La tabla
`predictions` no persiste `model_version`.

El threshold dinámico se calcula sobre la probabilidad calibrada inicial, pero
después se aplican ajustes GDP, demora estimada, demora de salida y ADS-B. La
etiqueta final usa la probabilidad ajustada con un cutoff calculado antes de los
ajustes. En una ejecución productiva observada, una tasa positiva cruda cercana
a 22.3% terminó en aproximadamente 39.5% final.

**Recomendación:** persistir modelo, calibrador, hash de schema y threshold por
predicción; aplicar la estrategia de decisión sobre la probabilidad final y
hacer que la API lea esa metadata en vez de inferirla desde configuración.

### F-08 — `/test-cases` puede evaluar predicciones posteriores a la salida

**Severidad:** alta
**Estado:** abierto

Los casos formales CP-01/CP-02 seleccionan la última predicción por vuelo sin
restringir explícitamente a `PRE_DEPARTURE`. Una predicción en ruta o posterior
al aterrizaje puede reemplazar la predicción genuinamente anticipada y mejorar
artificialmente AUC/Brier.

**Recomendación:** seleccionar la última predicción anterior a `actual_off_utc`
y reportar métricas por lead time: `>4h`, `2-4h`, `<2h` y en-route por separado.

### F-09 — `users.db` no es seguro con múltiples instancias

**Severidad:** alta
**Estado:** abierto

Cada instancia mantiene `/tmp/users.db` y sube el archivo completo a GCS luego
de cambios. Dos instancias pueden sobrescribirse entre sí. Además, el helper de
upload captura la excepción y los endpoints devuelven éxito aunque el cambio no
haya quedado persistido; al reciclar la instancia, el usuario o preferencia
puede desaparecer.

**Recomendación:** migrar usuarios a una base transaccional administrada
(Cloud SQL, Firestore u otra store con control de concurrencia) y no usar un
archivo SQLite sincronizado como storage de autenticación.

### F-10 — Suite API no reproducible desde un checkout limpio

**Severidad:** media
**Estado:** abierto

`tests/test_api.py` instancia `TestClient` y ejecuta login al importar el
módulo, sin entrar al context manager que dispara el lifespan. En un checkout
sin `users.db`, la recolección falla con `no such table: users`. Cinco tests se
omiten cuando la DB bundleada no contiene vuelos del día.

**Recomendación:** fixtures aislados para `users.db` y `live_data.db`, usar
`with TestClient(app)`, generar vuelos determinísticos y cubrir explícitamente
auth, administración, GCS refresh, `live_job` y conflictos de generación.

### F-11 — CLI ADS-B llama una función inexistente

**Severidad:** media
**Estado:** abierto

`feature_engineering_v7/adsb_lineage.py` invoca
`download_faa_registry()`, función no definida en el módulo. El CLI falla antes
de ejecutar el setup. El helper existente se llama `download_aircraft_db()`.

### F-12 — Índices `stable_id` ausentes en una DB nueva

**Severidad:** media
**Estado:** abierto

El schema fresco ya contiene la columna `stable_id`, pero la migración crea el
índice sólo cuando necesita agregar la columna. Una DB nueva queda sin
`idx_flights_stable`, `idx_predictions_stable` e `idx_actuals_stable`, aunque
las consultas live dependen de esos joins.

**Recomendación:** ejecutar siempre `CREATE INDEX IF NOT EXISTS` fuera del
condicional de migración y agregar una prueba sobre `PRAGMA index_list`.

### F-13 — Dependencias de artifacts sin pinning reproducible

**Severidad:** media
**Estado:** abierto

Los requirements permiten rangos amplios. Los calibradores serializados con
scikit-learn 1.8 se cargaron durante la auditoría con 1.9 y emitieron
`InconsistentVersionWarning`. Un cambio compatible a nivel de instalación no
garantiza compatibilidad del pickle ni resultados idénticos.

**Recomendación:** lockfile/constraints por imagen, manifest dentro del artifact
con versiones exactas y regeneración del artifact cuando cambia el runtime.

### F-14 — Warnings y errores silenciosos reducen observabilidad

**Severidad:** baja
**Estado:** abierto

- `ontimeai/lineage.py` convierte delays faltantes a `int64` y genera warnings
  de cast inválido.
- Varias rutas críticas capturan `Exception` y devuelven vacío o `None`, por lo
  que métricas, SHAP o features pueden degradarse sin un error visible.
- Ruff encontró imports/variables sin uso y `except/pass`; la mayoría son deuda
  de mantenimiento, no fallos productivos inmediatos.

**Recomendación:** logging estructurado, contadores de degradación por feature,
alertas de frescura/generación y eliminar silencios en rutas críticas.

## Orden recomendado de remediación

1. Rotar y externalizar secretos de F-01.
2. Build/deploy coordinado de la corrección F-02 en Backend y Scrapper.
3. Hacer atómico el refresh API de F-03.
4. Incorporar pruning y observabilidad de upload de F-04.
5. Unificar el contrato de features y corregir F-05/F-06.
6. Persistir trazabilidad y corregir threshold/evaluación de F-07/F-08.
7. Migrar usuarios y mejorar tests/dependencias según F-09 a F-14.

## Validación requerida después del deploy de F-02

1. Confirmar que ambos jobs de predicción y el harvester usan imágenes nuevas.
2. Verificar en logs que cada ciclo registra `base=<N>` y
   `generation=<N+1>` al subir.
3. Forzar una ejecución simultánea controlada: un escritor debe publicar y el
   otro registrar `generation conflict`, volver a descargar y completar sin
   sobrescribir la generación ganadora.
4. Descargar la DB final y ejecutar `PRAGMA quick_check` e `integrity_check`.
5. Confirmar que los datos producidos por ambos escritores aparecen en la
   generación final.
