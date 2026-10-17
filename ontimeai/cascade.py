"""Efecto cascada: cuánto de la demora de un vuelo llega a los siguientes.

Es la mitad del título de la tesis —*modelado de efecto cascada*— y el
anteproyecto lo compromete así: «dado un vuelo retrasado, estime el impacto en
vuelos subsiguientes que comparten aeronave o aeropuerto».

Este módulo cubre la **rotación de la aeronave**: el avión que llega tarde sale
tarde en su próxima pierna, y eso se arrastra. El efecto de aeropuerto
—congestión— es otro mecanismo y queda fuera; ver el final del docstring.

## Lo que dicen los datos

Medido sobre **779.777 pares de piernas consecutivas** del mismo avión en el
dataset BTS (2022-2024, todo EEUU), tomando sólo rotaciones de verdad —la
siguiente pierna sale del aeropuerto donde aterrizó la anterior— con holgura
programada de 20 minutos a 12 horas.

El hallazgo que no es obvio: **con turnaround ajustado la demora no se
traslada, crece.** Controlando dentro de cada tramo de holgura (con el avión en
horario la siguiente pierna sale 3 minutos ANTES en todos los tramos, así que
la base es plana y no hay efecto de selección):

    holgura        llegó tarde      salió tarde     se traslada
    20-45 min         36 min           49 min          136%
    45-70 min         37 min           46 min          124%
    70-100 min        39 min           38 min           97%
    100-180 min       46 min           19 min           41%
    180-300 min       72 min            5 min            7%

Estable en cinco ventanas de tres años y todas las estaciones: el tramo de
20-45 min va de 125% a 136%, el de 180+ de 5% a 7%.

Y ATL, que es donde opera el sistema, está en la zona mala: holgura mediana 67
minutos, con el 28% de las rotaciones entre 40 y 60 minutos y el 39% entre 60 y
90. Dos tercios de los turnarounds de ATL no absorben la demora.

## Por qué una tabla y no una fórmula

Se probaron dos formas cerradas y las dos fallan, medido fuera de ajuste sobre
390.045 pares:

    forma                            MAE    sesgo   p90 |err|
    fracción por tramo de holgura   16,7     -9,8       31,2
    capacidad fija en minutos       25,1     +9,1       56,0
    tabla medida en 2D              18,2     -0,1       30,7

La fracción tiene el MAE más bajo pero subestima siempre: cuando lo real son 96
minutos predice 59. La capacidad fija sobreestima: predice 318 donde lo real
son 157. La tabla no tiene sesgo y es la que acierta en las demoras grandes,
que son las que importan.

La razón de fondo es que la relación no es separable. Con 70-100 minutos de
holgura se absorben ~17 minutos sin importar si la demora que llega es de 20 o
de 180 —o sea cantidad fija, no porcentaje—, pero con holgura de 3 horas se
absorbe todo. Ninguna de las dos formas captura las dos cosas.

## Los límites, que importan para leer el número

- **Demoras de más de 4 horas**: la tabla sobreestima (179 contra 157 reales).
  Ahí la aerolínea interviene —cambia el avión, reprograma— y la cadena se
  rompe. El número se devuelve igual pero marcado como poco confiable.
- **Es una proyección, no un pronóstico**: sin sesgo, pero con 18 minutos de
  error absoluto medio. Sirve para decir «esta rotación va a arrastrar» y no
  para prometer una hora.
- **La demora de salida que devuelve es gate-out**, como la define BTS. NO es
  comparable con `actuals.departure_delay_min` cuando la fuente es FR24, que
  reporta despegue y trae +31 minutos de sesgo (ver Backend #11). Para
  comparar, usar demora de llegada, que sí es consistente entre proveedores.
- **Fuera de rango**: holguras de menos de 20 minutos no son turnarounds reales
  y las de más de 12 horas son un avión que durmió ahí, con la cadena cortada.

## Lo que queda afuera

El efecto de **aeropuerto** (un vuelo demorado que congestiona la pista y
atrasa a terceros sin relación de aeronave). Es otro mecanismo, mucho más
difícil de aislar de las demás causas, y se decidió encararlo aparte.

Tampoco hay modelo de secuencia (LSTM/Transformer) ni red neuronal de grafos,
que figuraban en la hoja de ruta original de este archivo. La tabla medida es
más simple, se puede defender ante un tribunal mostrando de dónde sale cada
número, y no había evidencia de que un modelo más grande hiciera falta.
"""
from __future__ import annotations

from dataclasses import dataclass

# Bordes de los tramos. El primero y el último acotan el rango con sentido:
# menos de 20 minutos no es un turnaround, más de 12 horas es otro día.
BORDES_HOLGURA_MIN = (20, 45, 70, 100, 180, 721)
BORDES_DEMORA_MIN = (0, 15, 30, 45, 70, 120, 240, 601)

# Demora media de salida de la pierna siguiente, en minutos.
# Filas: tramos de holgura. Columnas: tramos de la demora que llega.
# Cada celda tiene al menos 500 pares; la más chica, 500.
TABLA_PROPAGACION = (
    (12.9, 28.3, 44.5, 63.4, 99.1, 174.6, 355.8),  # holgura  20-45 min
    (9.3, 22.7, 38.3, 59.1, 95.9, 173.4, 349.8),   # holgura  45-70
    (5.7, 13.4, 24.8, 42.7, 76.7, 152.0, 325.0),   # holgura  70-100
    (5.2, 9.6, 12.1, 17.1, 33.3, 90.6, 250.5),     # holgura 100-180
    (5.0, 5.0, 5.9, 6.7, 8.4, 13.5, 77.4),         # holgura 180-721
)

# Minutos que un vuelo recupera en ruta, de la demora de salida a la de llegada.
# Medido sobre 279.311 vuelos (BTS, junio 2023): ARR = 1,006 * DEP - 4,5, con
# r=0,975. O sea una resta casi constante y no una pendiente: el colchón del
# block time es fijo. Por tramos recupera entre 2,5 y 4,3 minutos, y crece poco
# con la duración del vuelo (5 min en vuelos cortos, 9 en los de más de 4 h).
#
# NO confundir con el «pendiente 0,95, intercepto -29,8» que cita Backend #7:
# ese está ajustado sobre `departure_delay_min` de FR24, que mide despegue y no
# salida de puerta, y arrastra +31 minutos de sesgo (Backend #11). De ahí sale
# la afirmación de que un vuelo que sale 30 minutos tarde llega en horario, que
# los datos de BTS no sostienen: recupera 4 minutos.
RECUPERACION_EN_RUTA_MIN = 4.5

# Arriba de esto la aerolínea interviene y la cadena se rompe: la tabla
# sobreestima (179 contra 157 reales) y el resultado se marca poco confiable.
DEMORA_CADENA_ROTA_MIN = 240

# Por debajo de esto no se propaga nada, y no es una elección estética.
#
# La primera columna de la tabla (0-15 min) vale entre 5 y 13 minutos, pero eso
# NO es propagación: es la demora de salida que los vuelos tienen de base —con
# el avión en horario la siguiente pierna sale 3 minutos antes, y el promedio
# del lote ronda los 5 minutos tarde—. La medición no puede separar las dos
# cosas con esa granularidad.
#
# Encadenando sin este corte, esos 5 minutos se regeneran en cada paso y la
# cadena no muere nunca: una demora de 16 minutos terminaba arrastrando
# "demora" por toda la rotación. Lo encontró un test.
#
# Quince es además el umbral del DOT, el mismo con que se define el target.
UMBRAL_PROPAGACION_MIN = 15


def _indice(bordes: tuple[int, ...], valor: float) -> int:
    """Tramo al que cae `valor`, saturando en los extremos."""
    for i in range(len(bordes) - 1):
        if valor < bordes[i + 1]:
            return i
    return len(bordes) - 2


def demora_siguiente_pierna(holgura_min: float, demora_llegada_min: float) -> float:
    """Demora de salida esperada de la próxima pierna del mismo avión.

    `demora_llegada_min` es la demora de LLEGADA de la pierna anterior, que es
    la base consistente entre proveedores. Devuelve minutos de demora de
    salida, en semántica gate-out.

    Por debajo de `UMBRAL_PROPAGACION_MIN` devuelve cero: ahí la tabla sólo
    refleja la demora de salida de base, no propagación. La primera columna
    queda sin usar a propósito; se conserva porque es parte de la medición.
    """
    if demora_llegada_min <= UMBRAL_PROPAGACION_MIN:
        return 0.0
    if holgura_min < BORDES_HOLGURA_MIN[0]:
        # Menos de 20 minutos no es un turnaround real. Se usa el tramo más
        # ajustado, que es el que más propaga: subestimar acá sería el error
        # caro.
        holgura_min = BORDES_HOLGURA_MIN[0]
    i = _indice(BORDES_HOLGURA_MIN, holgura_min)
    j = _indice(BORDES_DEMORA_MIN, demora_llegada_min)
    return TABLA_PROPAGACION[i][j]


@dataclass(frozen=True)
class Pierna:
    """Una pierna de la rotación, con sus horarios programados en minutos UTC."""

    id_vuelo: str
    origen: str
    destino: str
    salida_programada_min: float
    llegada_programada_min: float


@dataclass(frozen=True)
class Proyeccion:
    """Lo que le llega a una pierna por la cadena."""

    id_vuelo: str
    origen: str
    destino: str
    orden: int
    holgura_min: float
    demora_salida_min: float
    demora_llegada_min: float
    cadena_rota: bool

    @property
    def demorado(self) -> bool:
        """Si cruza el umbral de 15 minutos, que es el target del modelo."""
        return self.demora_llegada_min > 15


def proyectar_rotacion(
    piernas: list[Pierna],
    *,
    demora_llegada_inicial_min: float,
    llegada_programada_inicial_min: float,
    max_piernas: int = 4,
) -> list[Proyeccion]:
    """Propaga una demora por las piernas siguientes del mismo avión.

    `piernas` van en orden y empiezan por la que sigue al vuelo demorado.
    `demora_llegada_inicial_min` y `llegada_programada_inicial_min` son la
    demora de llegada y la llegada programada de ESE vuelo: las dos hacen falta
    porque la holgura de la primera rotación se mide contra su llegada, no
    contra la salida de la pierna siguiente.

    Cada pierna recibe la demora de llegada proyectada de la anterior, con su
    propia holgura, así que una rotación con una escala larga en el medio corta
    la cadena sola. Sólo se devuelven las piernas que efectivamente reciben
    demora.

    Con holgura ajustada la demora se SOSTIENE en vez de morir: con 60 minutos
    de turnaround repetido, 16 minutos de demora se estabilizan en 18 y no
    bajan. No es un error de la proyección, es lo que significa el 124% de
    traslado medido en ese tramo. En la realidad la aerolínea rompe esa
    cadena cambiando el avión, y por eso `max_piernas` existe.

    `max_piernas` existe porque el error se acumula: la tabla tiene 18 minutos
    de error absoluto medio por paso, y a la cuarta pierna el intervalo es más
    ancho que el número. Cuatro ya es generoso.
    """
    salida: list[Proyeccion] = []
    demora_entrante = float(demora_llegada_inicial_min)
    llegada_anterior = float(llegada_programada_inicial_min)

    for orden, pierna in enumerate(piernas[:max_piernas], start=1):
        # El corte va ANTES de agregar: si lo que entra no propaga, esta pierna
        # no recibe nada y agregarla con cero sólo ensucia la salida con filas
        # que no dicen nada.
        if demora_entrante <= UMBRAL_PROPAGACION_MIN:
            break
        holgura = pierna.salida_programada_min - llegada_anterior
        dem_salida = demora_siguiente_pierna(holgura, demora_entrante)
        dem_llegada = max(0.0, dem_salida - RECUPERACION_EN_RUTA_MIN)
        salida.append(Proyeccion(
            id_vuelo=pierna.id_vuelo,
            origen=pierna.origen,
            destino=pierna.destino,
            orden=orden,
            holgura_min=holgura,
            demora_salida_min=dem_salida,
            demora_llegada_min=dem_llegada,
            cadena_rota=demora_entrante > DEMORA_CADENA_ROTA_MIN,
        ))
        demora_entrante = dem_llegada
        llegada_anterior = pierna.llegada_programada_min

    return salida
