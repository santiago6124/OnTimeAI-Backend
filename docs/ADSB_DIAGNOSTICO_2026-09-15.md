# Por qué se cortó el feed ADS-B

Criterio 1 del issue #9. Diagnóstico, y qué se hizo con cada parte.

## El síntoma

`aircraft_position` sin datos posteriores al **2026-08-12**, y al 15/09 con
**0 filas**: la purga de 30 días se llevó lo último que quedaba. Dos de las
cuatro etapas de la cadena de ajuste post-predicción llevaban ~34 días
recibiendo `None` en cada ciclo:

| columna de `predictions` | poblado |
|---|---|
| `adsb_eta_delay_min` | 0,1% |
| `adsb_holding_min` | 0,0% |

## La causa

Las dos fuentes están caídas, cada una por su motivo. Está en los logs del
harvester, en nivel `WARNING`, en cada corrida:

```
[WARNING] airplanes_live_client: airplanes.live: HTTP 403 —
  {"error": "Please contact us at contact@airplanes.live. Your email MUST
   include any links, a description of the project, and any information you
   deem appropriate."}

[WARNING] opensky_client: OpenSky: request failed —
  ConnectTimeoutError(host='opensky-network.org', port=443,
                      'Connection ... timed out. (connect timeout=15.0)')

[INFO] harvester: adsb_capture: both sources returned empty (likely network or rate)
```

**airplanes.live** no es un fallo técnico: bloquearon el acceso anónimo y piden
registrarse por mail describiendo el proyecto. Se resuelve escribiéndoles.

**OpenSky** da timeout de conexión, no 401 ni 403 — o sea que no llega a
negociar auth. Es consistente con bloqueo de rangos de datacenter; Cloud Run
sale por IPs de Google. El cliente soporta usuario y contraseña
(`OPENSKY_USERNAME` / `OPENSKY_PASSWORD`), hoy vacíos, pero credenciales no
arreglan un timeout de red.

Ninguna de las dos se resuelve desde el código. Quedan como acción para el
equipo.

## Lo grave no era el feed

El pipeline no falló. Las corridas figuran exitosas, no hay errores, y el
harvester loguea el problema en `WARNING` y sigue —que es la decisión correcta,
ADS-B no debe tumbar la cosecha—. Pero nada levantaba ese `WARNING`, así que el
sistema perdió dos señales durante un mes sin que nadie se enterara.

Eso es lo que se arregla acá, y aplica a cualquier fuente, no solo a ADS-B.

## Qué se agregó

**`/admin/db-stats` expone frescura por fuente.** Para cada tabla: cuándo
escribió por última vez, hace cuántos minutos, cuántos se toleran, y si eso ya
cuenta como caída.

| tabla | tolerancia | la alimenta |
|---|---|---|
| `predictions` | 60 min | live-pull (cada 15 min) |
| `actuals` | 120 min | harvester FR24 + AeroAPI |
| `weather_obs` | 180 min | IEM METAR (publica cada ~1 h) |
| `nas_status` | 180 min | NAS status FAA |
| `aircraft_position` | 120 min | ADS-B airplanes.live / OpenSky |

Las tolerancias son holgadas respecto de la cadencia real, para que un ciclo
lento no dispare ruido.

Una tabla **vacía** cuenta como caída: es exactamente el estado en que quedó
`aircraft_position`. Una tabla que **no existe** se omite, que es distinto —
significa que esa fuente nunca se instaló en esa base, no que dejó de escribir.

**El vigía alerta por fuente.** Cada una es su propia alerta, así que una caída
no queda tapada por el resto funcionando. Sigue avisando solo en los cambios de
estado: un problema que persiste no repite el aviso, y cuando se resuelve manda
la recuperación.

Solo pregunta por las fuentes si el backend responde. Con el backend caído cada
fuente daría un falso positivo y el aviso serían cinco alertas en vez de una.

Y el estado se arrastra entre corridas: una fuente que no se pudo evaluar
conserva lo que sabíamos de ella. Si su clave desapareciera, al volver el
backend se la daría por sana y volvería a avisar de algo que nunca se arregló.

## Lo que queda por hacer, fuera del código

- [ ] Escribir a contact@airplanes.live describiendo el proyecto
- [ ] Averiguar si OpenSky admite tráfico desde Cloud Run, o buscar salida por
      otra IP

Mientras tanto la alerta de ADS-B va a dispararse una vez y quedarse callada,
que es el comportamiento correcto para un problema conocido y persistente.
