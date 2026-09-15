# No promover `4year_v9_recal` — 2026-09-15

Decisión del issue #15. Se mide, no se promueve, y se explica por qué.

## Qué se comparó

Los dos artefactos comparten el mismo `model.lgb` (mismo tamaño, misma fecha) y
difieren solo en el calibrador:

- `4year_v9` → `Calibrator(isotonic)`
- `4year_v9_recal` → `StackedCalibrator`, que aplica `cal2(cal1(p))`

`cal1` resulta **idéntico** al calibrador de v9 — verificado numéricamente sobre
una grilla de 101 puntos, no por inspección. Eso vuelve la comparación barata:
la columna `predictions.proba_raw` ya guarda `cal1(booster)`, así que la salida
de v9_recal se obtiene aplicándole `cal2` encima. `cal2` es un sigmoide.

Hizo falta llegar hasta ahí porque los snapshots del training store, que sí
guardan la probabilidad cruda del booster, se vacían del outbox al entregarse a
GCS: quedaban 187 filas, todas demasiado recientes para tener resultado.

## El resultado

3.981 vuelos post-Fase 4 con resultado real, tasa real de retraso 7,7%:

| artefacto | Brier | AUC | ECE | p media |
|---|---|---|---|---|
| `4year_v9` | **0,0683** | 0,6787 | **0,0164** | **0,0694** |
| `4year_v9_recal` | 0,0710 | 0,6787 | 0,0490 | 0,0288 |

El AUC es idéntico porque ambos calibradores son monótonos: no cambian el orden.

Curva de calibración:

| bin | n | v9 dice | recal dice | real |
|---|---|---|---|---|
| 0,0-0,1 | 3.199 | 4,0% | 1,8% | 5,5% |
| 0,1-0,2 | 545 | **13,6%** | 5,2% | **12,7%** |
| 0,2-0,3 | 152 | **24,2%** | 8,7% | **21,1%** |
| 0,3-0,4 | 45 | **33,8%** | 12,2% | **28,9%** |
| 0,4-0,5 | 20 | 43,6% | 16,2% | 45,0% |

## Por qué

`4year_v9` ya está bien calibrado sobre esta cohorte: ECE 0,0164, y su
probabilidad media (6,94%) cae cerca de la tasa real (7,7%).

La recalibración de agosto corregía una sobre-confianza real **en ese momento**
—el `recal_report.json` reporta ECE 0,1259 antes y 0,0141 después, sobre 53.750
muestras—. Esa sobre-confianza ya no existe. Aplicar la corrección igual deja al
modelo infra-confiado: la probabilidad media cae a 2,88% contra una tasa real
de 7,7%, y el ECE empeora ×3.

Las dos diferencias entre aquella cohorte y esta explican el cambio: la
recalibración es anterior al switch a Fase 4, cuando la fuente de inferencia era
AeroAPI y no FR24, y la prevalencia rondaba el 20% contra el 7,7% de hoy.

## Qué hacer en su lugar

No regenerar la recalibración por ahora: con ECE 0,0164 no hay margen que
recuperar, y un calibrador nuevo ajustado sobre una prevalencia que se mueve
corre el mismo riesgo de quedar desfasado.

Lo que sí conviene es **monitorear el ECE**, para detectar cuándo vuelva a
abrirse en vez de descubrirlo un mes después. `scripts/analyze_adjustment_chain.py`
ya lo calcula; falta llevarlo a una métrica que el vigía pueda mirar.

## Reproducir

```bash
GCS_BUCKET=ontimeai-prod-live-db python3 scripts/compare_calibrators.py
```
