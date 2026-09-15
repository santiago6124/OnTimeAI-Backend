# Qué se purga, qué se conserva

Decisión del issue #12.

## El conflicto

`prune_db` borra a los 30 días. Eso mantiene la base acotada — necesario, ya
pesa 720 MB y #4 se queja de los ciclos — pero se lleva la materia prima de
tres issues del Frontend:

| issue | qué pide | qué había |
|---|---|---|
| Frontend #3 | 8 semanas de accuracy histórica | 4 semanas, y no crecía |
| Frontend #5 | serie para medir lead time advantage | se borraba a los 30 días de empezar |
| Frontend #11 | escenarios históricos para la demo | la ventana se corre sola |

Subir la retención empeora #4. Bajarla empeora esto. La salida es **separar lo
que se purga de lo que se conserva**.

## La decisión

**Lo crudo se sigue purgando a 30 días.** `predictions`, `prediction_shap`,
`flights`, `actuals`, `weather_obs`, `runs`, `harvester_runs`, `nas_status`,
`aircraft_position`.

**Los agregados se conservan sin límite.** `metrics_daily`, una fila por
(día, segmento):

```
~55 filas por día     (todos + ~30 aerolíneas + 24 horas)
~20.000 filas por año
```

Son kilobytes por semana contra gigabytes de lo crudo, así que el costo de
guardarlos para siempre es irrelevante.

`prune_db` borra con una **lista explícita** de `DELETE`, así que una tabla
nueva queda afuera sola; no hace falta excluirla. Un test lo fija: si alguien
agrega `DELETE FROM metrics_daily` a esa lista, falla.

## Qué se guarda por día

Por cada (día, segmento): `n_flights`, `n_delayed`, `n_flagged`, la matriz de
confusión completa, `auc`, `brier`, `ece`, `mean_proba`, `mean_threshold` y el
`model_version` que los produjo.

El `ece` cierra algo que había quedado pendiente en la decisión de #15: hasta
ahora no se monitoreaba, así que una deriva de calibración se descubría por
casualidad. Con la serie diaria se ve.

Segmentos: `all`, `carrier:XX`, `hour:HH`. **No por ruta** — ATL tiene cientos,
y multiplicarlas por día haría crecer la tabla más rápido de lo que justifica
lo que se consulta hoy.

## La unidad es un vuelo, no una predicción

`live_metrics.py` agrega sobre todas las filas de `predictions`. Un vuelo que
estuvo diez ciclos en la ventana pesa diez veces más que uno que entró en el
último, lo que sesga hacia los vuelos programados con más antelación.

`metrics_daily` toma **la última predicción anterior al aterrizaje** de cada
vuelo: la que el operador llegó a ver. El día es el de `scheduled_out_utc`, o
sea el día operativo, no el de la predicción.

## Cuándo se calcula

En cada ciclo del job, **antes de purgar** — la purga se lleva las filas de las
que salen los números.

Se recalculan los últimos **35 días**, más que la retención de 30 a propósito:
los `actuals` llegan tarde, y un día calculado antes de que sus vuelos settleen
queda con menos muestra de la que le corresponde. Recalcular la ventana entera
lo corrige solo.

Si el rollup falla, se loguea y el ciclo sigue. Perder un rollup cuesta un día
de histórico; frenar el pipeline cuesta la predicción de todos los vuelos.

## El agujero que se tapó de paso

`DELETE FROM flights WHERE scheduled_out_utc < ?` no alcanzaba a las filas
donde ese campo es NULL: en SQL `NULL < 'x'` es NULL, no verdadero. Esas filas
quedaban vivas para siempre. Y como los `actuals` se borran por orfandad contra
`flights`, cada vuelo inmortal mantenía vivo el suyo.

Medido el 15/09 sobre la base de producción:

```
flights con scheduled_out_utc NULL   : 16.520  (3,8%)
de esos, vistos hace más de 30 días  :  4.368
actuals más viejos que 30 días       :  2.040
de esos, colgados de un flight nulo  :  2.040  (100%)
```

**El 100%** de los `actuals` que sobrevivían a la retención colgaban de un
vuelo sin horario. Por eso la tabla arrastraba filas del 26/05 con una política
de 30 días.

Ahora esas filas se purgan por `first_seen_utc`, que es `NOT NULL` por esquema.

## Lo que queda abierto

**Frontend #5** (lead time advantage) necesita la serie de `estimated_out_utc`
versionada, que hoy no se guarda: `flights.estimated_out_utc` se sobrescribe en
cada actualización. Cuando se implemente, el lugar es una tabla propia fuera
del ciclo de purga, con la misma lógica que `metrics_daily`.

**Frontend #11** (escenarios de la demo) necesita congelar un día concreto
antes de que la purga lo alcance. Es un export puntual a JSON, no una tabla; se
hace cuando se elija el escenario.

## Consultar

```
GET /metrics/history?segment=all&days=56
GET /metrics/history?segment=carrier:DL&days=56
GET /metrics/history?segment=hour:17&days=56
```

Devuelve `precision`, `recall`, `actual_delay_rate`, `auc`, `brier`, `ece` y
`mean_threshold` por día. El default de 56 días son las 8 semanas de
Frontend #3.
