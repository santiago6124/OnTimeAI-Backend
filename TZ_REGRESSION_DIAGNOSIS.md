# Diagnóstico — TZ regression run 857 (100% NaN lineage)

**Estado**: investigación, no implementación de fix.

## Síntoma observado

Run 857 produjo predicciones con `mean proba ≈ 0.892` y `threshold quantile@0.22 = 0.974`. Esto es consistente con **100% NaN en lineage features** (sin señal de lineage el modelo extrapola alto vía proxies como `TAIL_DELAY_DECAY`).

Run 856 (anterior) tenía 59.6% NaN en lineage — alto pero parcial. El delta es el fix de TZ aplicado en `ontimeai/live.py:704-705`:

```python
df["EVENT_ORIGIN_UTC"] = pd.to_datetime(df["scheduled_off_utc"], errors="coerce", utc=True).dt.tz_localize(None).astype("datetime64[ns]")
df["EVENT_DEST_UTC"] = pd.to_datetime(df["scheduled_on_utc"], errors="coerce", utc=True).dt.tz_localize(None).astype("datetime64[ns]")
```

## Hipótesis principales

### H1 — Pollution del history por harvester rows sin actuals
Antes del fix, las filas FR24 con `+00:00` reventaban con `TypeError` o quedaban en `NaT` (según el path exacto). El sort de lineage (`ontimeai/lineage.py:51`) las descartaba implícitamente.

Después del fix, esas filas entran al sort con timestamps válidos pero **sin `arr_delay_min` (NULL)**. En `lineage.py:85` `settled_pos = np.where(np.isfinite(d))[0]` las excluye correctamente. PERO en `lineage.py:88-92`, `searchsorted(settled_pos, query_idx)` busca la cantidad de settled positions BEFORE `query_idx`. Si las filas no-settled del harvester se intercalan, `query_idx` aumenta pero `cnt_before` queda igual → el lineage **podría** quedar en NaN cuando antes lo hubiera resuelto.

**Probabilidad**: alta. Es coherente con el delta 856→857.

### H2 — Doble fila para el mismo physical flight
Tras el fix, la misma física flight tiene 2 filas (harvester + AeroAPI) con timestamps idénticos. En `lineage.py:51` `np.lexsort([t_orig, tail_codes])` los pone consecutivos pero su orden relativo es indeterminado. El "previous flight" puede ser elegido como una versión sin actuals → NaN.

**Probabilidad**: media. Mitigado por el dedup natural de stable_id pero NO completamente.

### H3 — Effect del Fix B (cross-source rescue) que acabo de aplicar
Mi propio fix de cross-source reconciliation, al recuperar más filas para history, **acentúa H1 y H2** si no hay dedup posterior. Esto debe validarse.

**Probabilidad**: relevante porque acabo de cambiar el código. Es un riesgo de regresión que el Fix B introduzca peor comportamiento si H1/H2 son ciertas.

### H4 — Bug específico de pandas en la conversión utc=True
`pd.to_datetime(["2026-05-22T22:00:00", "2026-05-22T22:00:00+00:00"], utc=True)` puede normalizar inconsistentemente cuando hay mezcla naive+TZ-aware. Si una serie tiene mayoría naive, pandas puede interpretar las naive como local (no UTC) y desplazar todo.

**Probabilidad**: baja. `utc=True` debería forzar UTC en ambos casos. Pero hay tickets pandas abiertos sobre comportamientos sutiles con mixed inputs.

## Evidencia que se necesita para confirmar

```sql
-- 1) ¿Cuántas filas en flights tienen scheduled_off_utc TZ-aware vs naive en los últimos 7d?
SELECT
  CASE WHEN scheduled_off_utc LIKE '%+00:00' THEN 'tz-aware' ELSE 'naive' END AS fmt,
  COUNT(*) n
FROM flights
WHERE scheduled_off_utc >= datetime('now', '-7 days')
GROUP BY 1;

-- 2) ¿Cuántas de cada formato tienen actuals matchea por stable_id?
SELECT
  CASE WHEN f.scheduled_off_utc LIKE '%+00:00' THEN 'tz-aware' ELSE 'naive' END AS fmt,
  COUNT(*) total,
  SUM(CASE WHEN a.arr_delay_min IS NOT NULL THEN 1 ELSE 0 END) settled
FROM flights f
LEFT JOIN actuals a ON a.stable_id = f.stable_id
WHERE f.scheduled_off_utc >= datetime('now', '-7 days')
GROUP BY 1;

-- 3) Casos donde existen 2 filas para el mismo (tail_num, scheduled_off_utc):
SELECT tail_num, scheduled_off_utc, COUNT(*) n
FROM flights
WHERE scheduled_off_utc >= datetime('now', '-7 days')
  AND tail_num IS NOT NULL
GROUP BY tail_num, scheduled_off_utc
HAVING COUNT(*) > 1
LIMIT 20;
```

## Fix candidato (NO aplicado — pendiente de evidencia)

Si H1 + H2 se confirman, el fix correcto es **dedup en `build_inference_frame` antes del sort de lineage**:

```python
# Pseudo-código a aplicar en ontimeai/live.py:680 (después del concat history+target)
# Dedup por (tail_num, EVENT_ORIGIN_UTC redondeado a minuto), keep first the one with non-NaN ARR_DELAY:
df["_dedup_min"] = pd.to_datetime(df["EVENT_ORIGIN_UTC"]).dt.floor("min")
df["_has_arr"] = df["ARR_DELAY"].notna().astype(int)
df = df.sort_values(["TAIL_NUM", "_dedup_min", "_has_arr"], ascending=[True, True, False])
df = df.drop_duplicates(subset=["TAIL_NUM", "_dedup_min", "_role"], keep="first")
df = df.drop(columns=["_dedup_min", "_has_arr"])
```

Riesgos del fix:
- Cambia el N de history → posibles cambios en otras features rolling (carrier_delay_rate_*, dest_delay_rate_*)
- Si los timestamps de harvester y AeroAPI difieren por >1min para el mismo physical flight, el dedup no agrupa
- Necesita test E2E antes de deployar

## Acción inmediata recomendada

1. Aplicar Fix A (lineage_fallback re-enable) y Fix B (cross-source rescue) primero
2. Re-correr el job en cloud → ver si lineage_hit_rate sube
3. Si run 858 tiene >50% NaN otra vez, ejecutar las queries de evidencia para H1/H2 sobre el DB
4. Solo entonces aplicar el dedup
