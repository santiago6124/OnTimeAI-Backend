# Duración del ciclo y frecuencia máxima

Cierre del issue #4, con lo medido el 2026-09-15. Está escrito para que nadie
tenga que volver a investigar de cero si el pipeline puede correr más seguido.

## Dónde se va el tiempo de un ciclo

Medido por fase desde dentro del job:

```
importar dependencias :    3,0 s
descarga de GCS       :    7,8 s
pipeline              :  176,2 s   ← clima 64 s + features/inferencia 91 s
agregados diarios     :    0,7 s
purga                 :    1,0 s
subida a GCS          :    8,9 s
                        ─────────
ciclo medido          :    3,1 min
ejecución de Cloud Run: 5,1 - 8,9 min
```

La diferencia —entre 2 y 6 minutos— transcurre **antes** de que el proceso
empiece a contar: aprovisionamiento del nodo, descarga de la imagen y arranque
del contenedor.

## Hipótesis descartadas, con el dato que las descartó

| hipótesis | medición |
|---|---|
| la base es muy grande y la transferencia domina | descarga 7,8 s + subida 8,9 s = **17 s** |
| los agregados diarios son caros | **0,6 - 0,7 s** |
| la purga es cara | **1,0 s** |
| importar pandas/lightgbm/sklearn es caro | **3,0 s** |
| el job reintenta tareas | `retriedCount` **vacío** en todas las ejecuciones |

Ninguna era el problema. El issue #4 afirmaba que *"con 710 MB, la transferencia
por sí sola domina el tiempo de ejecución"*; es empíricamente falso.

## Por qué no se pudo bajar el arranque en frío

Se recortó la imagen en dos pasos:

| | imagen | qué salió |
|---|---|---|
| original | 353 MB | |
| #54 | 320 MB | `live_data.db` bundleada, `fastapi`, `uvicorn`, `python-jose`, `api.py` |
| #55 | **247 MB** | `shap` → con él `numba` y `llvmlite`, 156 MB instalados |

**El efecto en el arranque no se pudo medir**, porque la caché de capas de los
nodos domina sobre el tamaño:

```
imagen 353 MB  → 120,4 s
imagen 320 MB  → 129,4 s
imagen 247 MB  → 323,6   291,6   239,0   292,7   125,1 s
```

Una imagen recién construida arranca **dos o tres veces más lento** durante las
primeras corridas, y recién después se estabiliza. Las cinco mediciones de la
imagen de 247 MB no bajan de forma ordenada —239 y después 292— pero terminan
en 125,1 s, que es el mismo valor que daban las imágenes de 320 y 353 MB.

Ahí está la conclusión: **con caché caliente, las tres imágenes tardan lo
mismo**. Recortar el 30% no cambió el arranque. Y durante las primeras corridas
después de cada despliegue el arranque se duplica o triplica, sea cual sea el
tamaño, lo que explica buena parte de la varianza de los ciclos (5,1 a 8,9 min)
sin necesidad de invocar otra causa.

Los dos recortes se dejan igual: valen por sí solos —menos que mantener, menos
que auditar— pero **no hay evidencia de que aceleren el ciclo**, y no conviene
afirmarlo. Tampoco la hay de que lo empeoren: la imagen de 247 MB, una vez
caliente, arranca igual que la de 353.

## El criterio que no se cumple

El issue pedía *"el ciclo vuelve a estar consistentemente por debajo de los 5
minutos"*. Hoy da 5,1 a 8,9.

Para lograrlo habría que atacar el pipeline —clima 64 s, features e inferencia
91 s— que es optimización real sobre código que funciona, o mudarse de Cloud
Run Jobs a algo con instancias tibias. Ninguna de las dos se justifica por lo
que este criterio protegía.

**Lo que protegía era el solapamiento**: que dos jobs escriban la misma base y
se pisen las subidas. Eso está cubierto por otras vías:

- el scheduler dispara cada 15 min y los ciclos duran 5-9: hay margen
- el vigía alerta cuando la mediana pasa los 10 min (dos tercios de la ventana)
- cada job toca la base solo 2-3 minutos de los 7 que dura, así que solaparse
  en reloj no implica chocar
- cero conflictos CAS medidos en 4 horas

Criterio propuesto en reemplazo: *"el ciclo no se acerca a la ventana del
scheduler, y hay alerta si lo hace"*.

## ¿Se puede correr más seguido?

**No sin colisiones.** La hora está completa:

```
live-pull     :00-:07   :30-:38        (6,1 - 8,3 min)
live-pull-2   :15-:21   :45-:52        (5,4 - 7,1 min)
harvester     :08-:14   :23-:28
              :38-:46   :53-:58        (4,6 - 7,9 min)
```

El margen más ajustado es de **un minuto**: el harvester de `:38`, en su peor
corrida medida, terminó `:46` mientras live-pull-2 arrancaba `:45`. No produjo
conflicto porque cada job toca la base al final de su corrida y no durante
toda ella, pero ese es el colchón que queda.

Subir la frecuencia exige primero acortar el ciclo. En ese orden.

## Un efecto secundario que conviene tener presente

Cada despliegue construye una imagen nueva, y las primeras corridas contra ella
arrancan dos o tres veces más lento. En un día con varios despliegues seguidos
—como el 15/09— eso infla los ciclos y puede parecer una degradación del
pipeline cuando es solamente caché fría. Antes de investigar un ciclo lento,
conviene mirar si hubo un despliegue reciente.

## Si alguien retoma esto

Lo que falta medir es el aprovisionamiento de Cloud Run separado de la descarga
de imagen. Desde dentro del proceso no se ve, y desde fuera la caché de capas
contamina la medición. Haría falta comparar contra un job de imagen mínima
—`python:3.11-slim` sin nada— corriendo en paralelo durante un rato largo.

Y conviene medir siempre con la imagen ya caliente: al menos cinco corridas
seguidas, descartando las primeras. Las conclusiones de una sola corrida en
este entorno no valen.
