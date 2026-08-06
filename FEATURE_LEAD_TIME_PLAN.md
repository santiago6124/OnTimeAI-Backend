# Plan de Feature: Comparación de Tiempo de Anticipación de Demoras (OnTimeAI vs. FlightRadar/AeroAPI)

Este documento detalla el diseño, la actualización de base de datos, la lógica de cálculo y las métricas para evaluar si OnTimeAI logra anticipar demoras antes que las fuentes tradicionales (AeroAPI/FlightRadar).

---

## 1. Cambios en la Base de Datos

### Modificación de la Tabla `predictions`
Actualmente, la tabla `predictions` no guarda el estado de los estimados externos del vuelo (`estimated_out_utc` / `estimated_in_utc`) en el instante de la predicción.
Para comparar con precisión, añadiremos estas columnas en [ontimeai/live.py](file:///Users/mateopappalardo/FACU/TEsis/OnTimeAI-Backend/ontimeai/live.py):

```sql
ALTER TABLE predictions ADD COLUMN api_estimated_out_utc TEXT;
ALTER TABLE predictions ADD COLUMN api_estimated_in_utc TEXT;
```

---

## 2. Cambios en el Proceso de Captura (Live Job)

### Modificación en `live_pull.py`
Al realizar cada predicción periódica en [live_pull.py](file:///Users/mateopappalardo/FACU/TEsis/OnTimeAI-Backend/live_pull.py), persistiremos el estimado de partida y llegada actual de la API en la tabla `predictions` junto con la probabilidad calculada por el modelo:

```python
pred_rows.append((
    df.loc[i, "fa_flight_id"], sid,
    pred_now, float(proba_adj), label_adj,
    float(threshold_used), threshold_strategy,
    proba_raw, float(gdp_orig), float(gdp_dest),
    int(atl_window), (float(carrier_smooth) if carrier_smooth is not None else None),
    (float(dep_delay) if dep_delay is not None else None),
    (float(adsb_delay) if adsb_delay is not None else None),
    (float(holding_min) if holding_min is not None else None),
    phase,
    df.loc[i, "estimated_out_utc"], # <-- Guardar estimado actual
    df.loc[i, "estimated_in_utc"],  # <-- Guardar estimado actual
))
```

---

## 3. Lógica de Evaluación y Scripts

Crearemos un nuevo script `scripts/compare_lead_times.py` para procesar el histórico de predicciones de los vuelos resueltos (con demoras reales $> 15$ minutos):

1. **Alerta del Modelo ($T_{\text{model}}$):** Primer timestamp en el que el modelo predijo una demora (`proba_delay >= threshold_used`).
2. **Alerta de la API ($T_{\text{api}}$):** Primer timestamp en el que el estimado externo del vuelo (`api_estimated_out_utc`) mostró un retraso $> 15$ minutos respecto a la hora programada (`scheduled_out_utc`).
3. **Ventaja de Tiempo ($\Delta T$):**
   $$\Delta T = T_{\text{api}} - T_{\text{model}}$$
   * Si $\Delta T > 0$: El modelo anticipó la demora $\Delta T$ minutos antes que la API.
   * Si $\Delta T < 0$: La API actualizó su estimado antes que el modelo.
   * Si la API nunca se actualizó hasta el despegue pero el modelo sí la anticipó: El modelo detectó un punto ciego de la API.

### Métricas Clave a Generar:
- **Win Rate del Modelo:** % de vuelos retrasados donde el modelo alertó antes que la API.
- **Ventaja de Tiempo Promedio (Lead Time):** Promedio de minutos de ventaja cuando el modelo gana.
- **Tasa de Puntos Ciegos de la API:** % de vuelos que la API nunca marcó con retraso pre-despegue, pero el modelo sí.
- **Gráfico de Anticipación:** Curva acumulada de detección según las horas restantes antes de la partida.
