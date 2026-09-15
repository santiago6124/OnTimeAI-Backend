# Semántica de las columnas de demora

`actuals` mezcla dos proveedores con definiciones distintas de lo mismo. Este
documento dice cuál es cuál, qué se puede comparar y qué no.

## El problema

`actuals.departure_delay_min` **no es comparable entre proveedores**.

| `source_provider` | qué mide | fórmula |
|---|---|---|
| `aeroapi` | demora de **puerta** | `actual_out − scheduled_out` |
| `fr24` | demora de puerta **más rodaje** | `actual_off − scheduled_out` |
| (nulo) | indeterminado | filas heredadas, sin forma de saberlo |

FR24 no expone gate-out. Lo dice su propio adaptador, en
[`fr24_client.py`](../../OnTimeAI-Scrapper/ontimeai_scrapper/fr24_client.py):

```python
"actual_out_utc": None,  # FR24 no expone gate-out, solo wheels-up
```

Así que su `departure_delay_min` arrastra el tiempo de rodaje, que en ATL no es
despreciable.

## La medida

Mediana de `departure_delay_min − arr_delay_min`, sobre las filas con ambas
presentes:

| `source_provider` | n | mediana |
|---|---|---|
| `fr24` | 410.650 | **+31,2 min** |
| `aeroapi` | 5.790 | +7,0 min |
| (nulo) | 14.217 | +17,0 min |

Los ~24 minutos de diferencia entre `fr24` y `aeroapi` son el rodaje de ATL.

Se confirma por otro lado: `actuals.actual_out_utc` está poblado en solo 3,2% de
las filas y no avanza desde el 2026-08-14 — solo lo llena AeroAPI, y desde Fase
4 el 96% de la muestra viene de FR24.

## Qué usar

**Para demora de llegada**: `arr_delay_min` es comparable entre proveedores.
Ambos la calculan como `actual_in − scheduled_in`. Es la base del target del
modelo (`arr_delay_min > 15`) y la que conviene usar para cualquier análisis.

**Para demora de salida comparable**: filtrar `source_provider = 'aeroapi'`. Es
la única con semántica de puerta. El costo es cobertura: queda el 1,3% de la
muestra.

**No existe** una demora de puerta comparable que cubra toda la base. Derivarla
restando una estimación de rodaje por aeropuerto y hora es posible, pero
introduce un error propio; mientras no haga falta, no se hizo.

## Quién la consume hoy

`live_pull.load_gate_departure_delays()` alimenta
`intermediate_dep_delay_adjust`, cuyas bandas están validadas contra BTS para
demora de **puerta**:

```
dep_delay  5..15 → p 0.25      15..30 → p 0.75
          30..60 → p 0.90        > 60 → p 0.97
```

Hasta el 2026-09-15 esa consulta no filtraba por proveedor. Un vuelo que
empujaba en horario y rodaba media hora en ATL entraba en la banda 30-60 y su
probabilidad saltaba a 0,90. Ahora filtra `source_provider = 'aeroapi'`.

Ninguna feature del modelo v9 consume esta columna: las 84 del booster salen de
`prepare_inference_frame()`, y la demora de salida entra solo por el ajuste
post-predicción de arriba.

## Antecedente

Apareció al ajustar el modelo de propagación en cascada: devolvía un intercepto
espurio de ~24 min hasta cambiar la base a demora de llegada.

Ver issue #11.
