# Pruebas de stress y esfuerzo sostenido con k6

Scripts en `k6/`, workflow en `.github/workflows/stress.yml`, resultados de la
prueba de concepto en `k6/resultados/`. Herramienta: [k6](https://k6.io)
(Grafana), que ejecuta escenarios escritos en JavaScript y falla con código 99
cuando se cruza un umbral declarado.

## Qué se prueba

La API de FastAPI (`api.py`) levantada con **un solo proceso de uvicorn** —el
mismo `CMD` del `Dockerfile`, y la misma forma en que corre en Cloud Run— sobre
una **base fixture**: una copia del `live_data.db` versionado con 200 vuelos de
hoy sembrados por `scripts/make_fixture_db.py` (la siembra vive en
`ontimeai/semillas.py` y es la misma que usa el fixture de `tests/test_api.py`).
El snapshot del repo no tiene vuelos de hoy; sin la siembra `/flights`
devuelve `[]` y nada de lo que cuelga de un vuelo se ejercita.

Cada usuario virtual (VU) hace login una vez y después repite el **mix de un
usuario del dashboard**, nueve requests por iteración con un segundo de pausa:

| Request | Qué hace la API |
|---|---|
| `GET /flights` | la ventana activa: una consulta con `ROW_NUMBER()` sobre todas las predicciones |
| `GET /flights/{id}` | detalle de un vuelo al azar de la lista; recalcula la ventana activa y lee el SHAP cacheado |
| `GET /flight-history/{id}` | historial de predicciones del vuelo |
| `GET /cascade/{id}` | propagación por rotación (encuentra la pierna siguiente sembrada) |
| `GET /metrics/summary`, `/metrics/hourly` | agregados del día sobre la misma ventana |
| `GET /metrics/model` | metadatos del artefacto + AUC/Brier vivos con pandas y scikit-learn |
| `GET /weather/ATL` | último METAR de la tabla `weather_obs` |

Cada request lleva la etiqueta `endpoint` y alimenta una métrica `Trend`
propia (`latencia_<endpoint>`), así el resumen tiene p95 por endpoint sin tener
que declarar un umbral sobre cada uno.

## Por qué no se prueba producción

El backend de Cloud Run está fijado a **una instancia** (`--max-instances 1`,
porque `users.db` es SQLite en `/tmp`), con 2 vCPU que comparte con el refresco
de la base de ~700 MB, que ocurre dentro de un request. Una rampa a 50 VUs lo
dejaría sin servicio para los usuarios reales durante la prueba, y lo que se
mediría sería la red y el autoscaler, no la API. Además la base de producción
cambia cada diez minutos: ninguna medición sería reproducible. Por eso la
prueba corre contra la API levantada en el propio runner, con una base que se
construye igual cada vez (`--semilla` fija).

## Escenarios

| Escenario | Carga | Duración | Qué pregunta responde |
|---|---|---|---|
| `smoke.js` | 1 VU | 30 s | ¿Login y los ocho endpoints responden 200 con JSON? Es la condición previa de los otros dos. |
| `stress.js` | rampa 0 → 20 → 50 → 0 VUs | 3 × 1 min | ¿Dónde empieza a degradar, y degrada sin fallar (sin 5xx, sin respuestas rotas)? |
| `soak.js` | 20 VUs constantes | 10 min | ¿La latencia se mantiene estable en el tiempo o crece (fugas, conexiones, caches)? Compara el p95 de la iteración completa del primer minuto con el del último. |

Las duraciones se pueden acortar por entorno (`SMOKE_DURATION`,
`STAGE_DURATION`, `SOAK_DURATION`, `SOAK_VUS`) para una prueba de concepto;
los defaults son los de la tabla.

## Umbrales y por qué

En los tres escenarios:

- `http_req_failed: rate<0.01` — menos del 1 % de respuestas con error (k6
  cuenta como error los 4xx/5xx y los timeouts, que por defecto son de 60 s).
- `checks: rate>0.99` — al menos el 99 % de los checks (status 200 y cuerpo
  JSON) pasan.

En stress y soak, además, un techo sobre el p95 global de `http_req_duration`:

| Escenario | p95 medido en la PoC local (dos corridas) | Umbral | Variable |
|---|---|---|---|
| stress | 14,6 s y 11,1 s | **30 s** | `P95_MAX_MS` |
| soak (3 min) | 17,5 s y 6,8 s | **35 s** | `P95_MAX_MS` |

La regla es **≈ 2× la peor medición local, redondeado**: los runners de GitHub
son más lentos que la máquina de desarrollo y un umbral ajustado al milímetro
fallaría por el entorno, no por la API. Se toma la peor corrida y no la mejor
porque las dos difirieron bastante con la misma API y la misma fixture (ver la
PoC): una laptop de desarrollo no es un entorno controlado.

Hay que leer esos números por lo que son. **No son objetivos de servicio**: a
50 VUs la API no da latencias interactivas y nadie debería esperar que las dé
con un solo proceso. Lo que la PoC mostró —abajo— es que la API sirve unas
6,5 req/s de este mix y que cualquier carga por encima de 1-2 VUs la satura:
los requests hacen cola y la latencia crece de forma lineal con los VUs, pero
**sin un solo error**. Los umbrales protegen entonces tres cosas: que siga sin
fallar bajo sobrecarga, que no aparezcan timeouts, y que la cola no se vaya
más allá del doble de lo conocido (una regresión de rendimiento grosera, del
tipo "una consulta nueva recorre toda la tabla").

Si un run de CI falla por el p95 y la máquina es la causa —misma versión de la
API, mismo fixture—, se ajusta el umbral y se anota acá con la medición del
runner. Si falla por errores o checks, es la API.

## Cómo correr en local

Hace falta `k6` (`brew install k6`) y el venv del backend.

```bash
# 1. La base fixture (85 MB; se sobreescribe si existe)
python scripts/make_fixture_db.py --destino /tmp/fixture.db --vuelos 200

# 2. La API, apuntando a esa copia y a un users.db temporal
DB_PATH_OVERRIDE=/tmp/fixture.db USERS_DB_PATH_OVERRIDE=/tmp/users.db \
JWT_SECRET_KEY=test-only-secret-not-used-in-production \
API_USERNAME=test-admin API_PASSWORD=test-only-password \
API_USERNAME_VIEWER=test-viewer API_PASSWORD_VIEWER=test-only-password-viewer \
uvicorn api:app --port 8000 --log-level warning

# 3. Los escenarios (en otra terminal)
k6 run -e API_USERNAME=test-admin -e API_PASSWORD=test-only-password k6/smoke.js
k6 run -e API_USERNAME=test-admin -e API_PASSWORD=test-only-password k6/stress.js
k6 run -e API_USERNAME=test-admin -e API_PASSWORD=test-only-password -e SOAK_DURATION=3m k6/soak.js

# 4. La tabla
python scripts/k6_resumen.py k6/resultados/*-summary.json
```

`BASE_URL` cambia el destino (default `http://localhost:8000`). Las
credenciales son las de prueba de `tests/conftest.py`; sólo existen en el
`users.db` temporal que crea el lifespan. `GCS_BUCKET` tiene que estar sin
definir: con él la API intentaría refrescar la base desde el bucket.

`DB_PATH_OVERRIDE` y `USERS_DB_PATH_OVERRIDE` son las dos variables que se
agregaron a `api.py` para esto. Sin ellas el comportamiento es el de siempre
(bundleado sin `GCS_BUCKET`, `/tmp` con él). No se reutilizó `DB_PATH` porque
`ontimeai/live.py` ya lee esa variable con otro significado para los jobs.

## Cómo leer los resultados

- **Al final de cada corrida** k6 imprime el resumen de `k6/lib.js`: duración,
  VUs, iteraciones, req/s, p50/p95/p99 global, errores, checks, p95 por
  endpoint y el veredicto de cada umbral. El soak agrega la deriva (p95 de la
  iteración completa en el primer y en el último minuto).
- **`k6/resultados/<escenario>-summary.json`** es el `handleSummary` completo
  (`SUMMARY_PATH` lo cambia). Es lo que lee `scripts/k6_resumen.py` para
  armar la tabla Markdown.
- **En CI**, la tabla queda en el *step summary* del run y los JSON más el
  log de uvicorn se suben como artifact (`k6-resultados-<run>`). Si un paso de
  k6 falla con exit 99, el job falla; el resumen y el artifact se publican
  igual.
- Los JSON fechados `2*-poc-local-*` de `k6/resultados/` están versionados
  como evidencia de la PoC; el resto de esa carpeta está en `.gitignore`.

## Prueba de concepto local — 2026-10-23

Entorno: Mac con Apple Silicon, Python 3.12, un proceso de uvicorn, k6 v2.3.0,
fixture de 200 vuelos (semilla 2026). El soak se acortó a 3 minutos con
`SOAK_DURATION=3m`; el default del script sigue en 10.

Corridas finales, con los umbrales definitivos (los JSON están en
`k6/resultados/2026-10-23-poc-local-*-summary.json`; la tabla la genera
`scripts/k6_resumen.py`):

| Escenario | VUs máx | Iteraciones | req/s | p50 ms | p95 ms | p99 ms | Errores HTTP | Checks | Umbrales |
|---|---|---|---|---|---|---|---|---|---|
| smoke | 1 | 20 | 5.8 | 37 | 135 | 325 | 0.00 % | 100.00 % | OK (2/2) |
| stress | 50 | 142 | 7.1 | 1696 | 11123 | 12563 | 0.00 % | 100.00 % | OK (3/3) |
| soak | 20 | 163 | 6.7 | 2147 | 6755 | 10695 | 0.00 % | 100.00 % | OK (3/3) |

Curva de capacidad con todas las corridas del día (45 s a 3 y 10 VUs con
`soak.js`; stress y soak se corrieron dos veces, la primera para medir y la
segunda con los umbrales ya fijados):

| VUs | req/s | p50 | p95 | p99 | Errores |
|---|---|---|---|---|---|
| 1 (smoke) | 5,8 | 37 ms | 135 ms | 325 ms | 0 |
| 3 | 6,0 | 372 ms | 943 ms | 1,4 s | 0 |
| 10 | 6,6 | 859 ms | 3,6 s | 4,0 s | 0 |
| 20 (soak, 3 min, 1.ª corrida) | 3,3 | 4,9 s | 17,5 s | 27,3 s | 0 |
| 20 (soak, 3 min, 2.ª corrida) | 6,7 | 2,1 s | 6,8 s | 10,7 s | 0 |
| 0 → 50 → 0 (stress, 1.ª corrida) | 6,9 | 1,3 s | 14,6 s | 16,0 s | 0 |
| 0 → 50 → 0 (stress, 2.ª corrida) | 7,1 | 1,7 s | 11,1 s | 12,6 s | 0 |

Lo que dice:

1. **La capacidad es de ~6,5 req/s** de este mix y no depende de los VUs: con
   1 VU ya se alcanza (5,8 req/s con un segundo de pausa por iteración). A
   partir de ahí sólo crece la cola: la latencia sube con los VUs (p95 de
   0,9 s a 3, 3,6 s a 10, 7-17 s a 20, 11-15 s en la rampa a 50). El proceso
   único serializa: FastAPI despacha los endpoints sincrónicos a un pool de
   hilos, pero cada uno convierte filas a dicts y serializa JSON bajo el GIL.
2. **Las dos corridas a 20 VUs difirieron 2,6×** en throughput (3,3 contra
   6,7 req/s) y en p95 (17,5 contra 6,8 s) con la misma API y la misma
   fixture. La primera vino justo después de la corrida de 10 VUs; es la
   razón de tomar la peor medición para los umbrales y de no comparar números
   de la Mac con los del runner.
3. **Dónde está el costo.** A 1 VU, los endpoints livianos responden en 1-2 ms
   (`/weather`, `/cascade`, `/flight-history`); los pesados en 110-220 ms
   (`/flights`, `/flights/{id}`, `/metrics/summary`, `/metrics/hourly`,
   `/metrics/model`). Los cuatro primeros recalculan la ventana activa
   (`_latest_predictions_active`) en cada request, sin ningún cache, sobre
   las 45.741 predicciones del snapshot; `/metrics/model` suma pandas y
   scikit-learn. En producción la base tiene cuatro veces más predicciones, así
   que la API real es más lenta por request que esta fixture.
4. **Degrada sin fallar.** Ningún escenario produjo un error HTTP ni un check
   fallido: la API hace cola, no se cae. Eso es lo que los umbrales de errores
   y checks fijan como piso.
5. **No hay deriva en el soak.** El p95 de la iteración completa fue de 27,2 s
   en el primer minuto y de 29,4 s en el último (en la primera corrida, 70 s y
   56 s): la cola se llena y se queda ahí, sin señal de fuga. Con tres minutos
   la comparación es gruesa; los diez del default la hacen más útil.

Qué haría falta para subir la capacidad —fuera del alcance de este cambio—:
cachear la ventana activa unos segundos (los datos cambian cada 10-30 min),
no recalcularla en `/flights/{id}`, y evaluar varios workers de uvicorn (hoy
lo complica `users.db` en `/tmp` y el refresh de la base dentro del proceso).

## Corridas en CI — 2026-10-23

Runner `ubuntu-latest` (4 vCPU), Python 3.11, un proceso de uvicorn, misma
fixture de 200 vuelos. Las tablas salen del step summary de cada run; los JSON
están en el artifact `k6-resultados-<n>`.

Run del `pull_request` del PR #95 (smoke + stress):
<https://github.com/santiago6124/OnTimeAI-Backend/actions/runs/40011348475>

| Escenario | VUs máx | Iteraciones | req/s | p50 ms | p95 ms | p99 ms | Errores HTTP | Checks | Umbrales |
|---|---|---|---|---|---|---|---|---|---|
| smoke | 1 | 16 | 4.7 | 60 | 190 | 197 | 0.00 % | 100.00 % | OK (2/2) |
| stress | 50 | 502 | 22.5 | 473 | 3299 | 4017 | 0.00 % | 100.00 % | OK (3/3) |

Run disparado a mano desde la rama (`gh workflow run stress.yml --ref
feat/stress-k6`), con el soak completo de 10 minutos:
<https://github.com/santiago6124/OnTimeAI-Backend/actions/runs/40011382323>

| Escenario | VUs máx | Iteraciones | req/s | p50 ms | p95 ms | p99 ms | Errores HTTP | Checks | Umbrales |
|---|---|---|---|---|---|---|---|---|---|
| smoke | 1 | 16 | 4.7 | 60 | 190 | 200 | 0.00 % | 100.00 % | OK (2/2) |
| stress | 50 | 496 | 22.2 | 518 | 3231 | 3814 | 0.00 % | 100.00 % | OK (3/3) |
| soak | 20 | 1788 | 23.7 | 607 | 1685 | 1918 | 0.00 % | 100.00 % | OK (3/3) |

Deriva del soak en el runner: p95 de la iteración completa de **6,7 s en el
primer minuto y 7,4 s en el último** (+10 %) a lo largo de 10 minutos y 1.788
iteraciones, sin errores. No hay señal de fuga ni de degradación acumulativa.

El runner dio **3× más throughput y un p95 3,4× menor** que la Mac en el
stress (22,5 contra 7,1 req/s; 3,3 contra 11,1 s), y en el soak a 20 VUs
23,7 req/s con p95 de 1,7 s contra 6,7 req/s y 6,8 s, con la misma API y la
misma fixture. La hipótesis es que en Linux los hilos del pool aprovechan los
núcleos mientras SQLite tiene el GIL liberado y en macOS no; no se investigó
más porque la prueba mide la API en el entorno donde corre el gate. La
consecuencia práctica es que los umbrales de p95 —fijados sobre la peor
medición local— quedan a ~9× el p95 del runner. Son seguros como gate de
"no empeoró al doble"; si con más corridas acumuladas se quiere uno más fino
para CI, bajar `P95_MAX_MS` a ~2× el p95 del runner y anotarlo acá.

## Limitaciones conocidas

- El soak **no corre en los PR** (sumaría 10 minutos a cada uno); corre en
  cada push a `main` y por `workflow_dispatch`, que también se puede disparar
  sobre una rama (`gh workflow run stress.yml --ref <rama>`). Si su p95 en el
  runner supera los 35 s, aplicar la regla de arriba y anotarlo.
- Los números del runner y los de la Mac no son comparables entre sí; lo
  comparable es cada entorno consigo mismo entre corridas.
- La fixture es el snapshot del repo (67.559 vuelos, 45.741 predicciones);
  producción tiene unas cuatro veces más. La prueba mide la API, no la base.
