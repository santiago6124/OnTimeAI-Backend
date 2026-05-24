# Hallazgos — Desbloqueo del live-predict (2026-05-24)

Sesión de debug que partió de un pedido simple ("hagamos una prueba de predecir un
vuelo en vivo") y terminó destapando **4 bugs distintos** que mantenían el pipeline
de predicciones caído desde 2026-05-21. **El diagnóstico previo en
`OnTimeAI-Scrapper/SESSION_HALLAZGOS.md §4.4` era incorrecto** — lo que se
documentó como "LightGBM SIGSEGV en `predict_proba`" en realidad eran tres bugs
encadenados que solo se podían ver al irlos destapando.

---

## 1. Resumen ejecutivo

| Item | Estado |
|---|---|
| Bug 1 — `TypeError: boolean value of NA is ambiguous` | ✅ FIXED |
| Bug 2 — `model.lgb` con CRLF line endings → SIGSEGV en load | ✅ FIXED (la raíz del SIGSEGV documentado) |
| Bug 3 — `conn` no flusheaba antes del upload GCS | ✅ FIXED |
| Bug 4 — Race condition Backend vs Harvester en bucket DB | 🟡 Mitigado (harvester pausado), arquitectura pendiente |
| Predicciones reales generadas | ✅ 52 vuelos en run 856 (2026-05-24T19:11:33 UTC) |
| Scheduler `ontimeai-pull-scheduler` | 🔴 PAUSED (sin reactivar) |
| Scheduler `ontimeai-harvester-scheduler` | 🔴 PAUSED (sin reactivar — race condition) |

**Lo que se demostró:** el pipeline end-to-end funciona. Backend descarga DB,
features, ML predict, persiste, sube. 52 predicciones reales escritas con
quantile@0.22 threshold=0.5810, distribución coherente con el patrón histórico
(arrivals concentran el riesgo, departures de ATL casi todas on-time).

**Lo que NO se demostró:** que el sistema pueda correr automáticamente. El
race condition entre los dos jobs sigue activo y va a sobrescribirse mutuamente
cuando reactivemos los schedulers.

---

## 2. Punto de partida y diagnóstico inicial fallido

`SESSION_HALLAZGOS.md §3.5 / §4.4` decía:

> **Hallazgo F4.4 (PREEXISTENTE, BLOQUEANTE) — LightGBM SIGSEGV en step [5] predict**
> - Síntoma: miles de warnings `[LightGBM] [Fatal] Model format error...`
> - Crash con signal 11 durante `predict_proba` o el calibrator
> - Hipótesis pendientes de testing:
>   1. Corrupción de memoria en LightGBM por feature dtype contaminado
>   2. Mismatch numpy/pandas en el image vs versiones de entrenamiento
>   3. `lineage_fallback.joblib` deserializa pero introduce estado inválido
>   4. Feature `cat_mapping` retorna índice fuera de rango

**Ninguna de las 4 hipótesis era correcta.** El SIGSEGV ocurría en
`lgb.Booster(model_file=...)` — **al cargar el booster**, no al predecir. Pero
ese síntoma quedó enmascarado por el bug NA que crasheaba antes.

---

## 3. Cómo se destapó cada bug (orden secuencial)

### 3.1 — Tick 1: bug NA (no era SIGSEGV)

Ejecutamos `gcloud run jobs execute ontimeai-live-pull` (execution `x8dsm`).
Esperábamos ver el SIGSEGV documentado. Lo que vimos fue **otro** crash:

```
File "/app/ontimeai/live.py", line 632, in build_inference_frame
    df["FLOW_ATL"] = np.where(df["ORIGIN"].eq("ATL"), "DEP_FROM_ATL", "ARR_TO_ATL")
TypeError: boolean value of NA is ambiguous
```

**Causa:** el harvester FR24 captura algunos vuelos privados (NetJets `EJA409`,
vuelos `WN` mal-tipeados sin origin) con `origin` o `dest` NULL. Cuando
`build_inference_frame` hace JOIN con `actuals` para hidratar el lineage history,
trae esas filas al inference frame. En pandas 3.x, `Series.eq()` ya no coerce
NA → False y `np.where()` no maneja el NA.

```sql
SELECT COUNT(*) FROM flights f JOIN actuals a ON a.stable_id = f.stable_id
WHERE f.stable_id IS NOT NULL AND (f.origin IS NULL OR f.dest IS NULL);
-- → 7 filas (todas privados / mal-tipeados FR24)
```

**Fix** (`ontimeai/live.py:632`):

```python
# Antes
df["FLOW_ATL"] = np.where(df["ORIGIN"].eq("ATL"), "DEP_FROM_ATL", "ARR_TO_ATL")
df["PAR_AIRPORT"] = np.where(df["ORIGIN"].eq("ATL"), df["DEST"], df["ORIGIN"])

# Después
_is_atl_origin = df["ORIGIN"].fillna("").eq("ATL")
df["FLOW_ATL"] = np.where(_is_atl_origin, "DEP_FROM_ATL", "ARR_TO_ATL")
df["PAR_AIRPORT"] = np.where(_is_atl_origin, df["DEST"], df["ORIGIN"])
```

### 3.2 — Tick 2: ahora SÍ apareció el SIGSEGV (execution `m2m4r`)

Con el fix NA aplicado, `build_inference_frame` terminó OK. El log mostró:

```
[5] Building features and predicting...
   48 target rows | 19273 history rows for lineage
[LightGBM] [Warning] Model format error, expect a tree here. met split_gain=67.7922 ...
[LightGBM] [Fatal] Model format error, expect a tree here. met 2151684608 1048704 ...
... (cientos de líneas) ...
Container terminated on signal 11.
```

Los prints de mi diagnostic (`lineage fallback disabled` y `loaded cold-deck
fallback`) **no aparecieron**. Eso ubicó el crash entre línea 247 (`print(target
rows)`) y línea 252 (`fallback_path = ...`). El único call en ese rango es
`meta = load_artifact(args.artifact)`.

**Conclusión:** el SIGSEGV ocurre en `lgb.Booster(model_file=...)`, no en
`predict_proba` como decía el HALLAZGOS previo. Cualquier hipótesis sobre features
contaminadas, fallback, cat_mapping era irrelevante — el modelo nunca llega a ser
usado.

### 3.3 — Intento fallido: bypass fallback + single-thread + LGB pin

Aplicamos las recomendaciones de `SESSION_HALLAZGOS.md §8.1`:
- `DISABLE_LINEAGE_FALLBACK=1` env var → no cambió nada (predict nunca ejecuta)
- `OMP_NUM_THREADS=1`, `LIGHTGBM_NUM_THREADS=1` → no cambió nada
- `lightgbm==4.5.0` (pin más conservador que 4.6) → no cambió nada
- `params={"num_threads": 1}` en `lgb.Booster()` → no cambió nada

Esto descartó: race condition de threads, versión de LightGBM, fallback corrupto.

### 3.4 — Tick 3: la causa raíz era CRLF en `model.lgb`

Inspección del archivo binario:

```
$ od -c artifacts/4year_v9/model.lgb | head -5
0000000   t   r   e   e  \r  \n   v   e   r   s   i   o   n   =   v   4
0000020  \r  \n   n   u   m   _   c   l   a   s   s   =   1  \r  \n   n

$ file artifacts/4year_v9/model.lgb
artifacts/4year_v9/model.lgb: ASCII text, with very long lines (10884), with CRLF line terminators

$ grep -c $'\r' artifacts/4year_v9/model.lgb
42215
```

**El archivo `model.lgb` se serializó con line terminators CRLF (Windows-style)**
en algún punto del workflow de entrenamiento que corrió en una máquina Windows.
LightGBM 4.x parsea `model.lgb` esperando LF como terminator. Cuando encuentra
`key=value\r\n`:
- Para metadata (`num_class=1\r`) el `\r` queda en el valor → parser interpreta
  `1\r` como un int de 2 bytes y sigue mal alineado.
- Cuando intenta leer la siguiente sección espera `Tree=N` pero encuentra
  `split_gain=...` o bytes shifted — emite "Model format error, expect a tree
  here" y trata de avanzar.
- Eventualmente accede memoria fuera del buffer parseado → SIGSEGV.

Los "miles de warnings" con bytes random NO son el modelo en uso — es LightGBM
intentando saltar de sección en sección en un texto donde TODAS están corruptas
por el `\r` al final de cada `key=value`.

**Fix:**

```bash
# Backup + convertir CRLF → LF (aplicado a ambos artifacts)
cp artifacts/4year_v9/model.lgb artifacts/4year_v9/model.lgb.crlf.bak
tr -d '\r' < artifacts/4year_v9/model.lgb.crlf.bak > artifacts/4year_v9/model.lgb
cp artifacts/4year_v9_recal/model.lgb artifacts/4year_v9_recal/model.lgb.crlf.bak
tr -d '\r' < artifacts/4year_v9_recal/model.lgb.crlf.bak > artifacts/4year_v9_recal/model.lgb
```

`.gitignore` actualizado con `*.crlf.bak` para no commitear los backups.

**¿Por qué pasó esto y antes funcionaba?**
- En la sesión documentada (HALLAZGOS.md), las predicciones del backend dejaron
  de funcionar el 2026-05-21. La causa fue identificada erróneamente.
- Hipótesis sobre origen del CRLF: algún re-save del modelo desde Windows
  Notepad o git con `core.autocrlf=true`. El archivo en repo quedó con CRLF.
- LightGBM 4.x es más estricto con el formato que versiones anteriores. Las
  versiones 3.x probablemente toleraban CRLF.

Tras el fix, la execution `t2slg` corrió successfully: `wrote 49 predictions
... Done. run_id=856 ... exit(0)`.

### 3.5 — Tick 4: predicciones escritas pero NO aparecen en GCS

A pesar de "wrote 49 predictions" y "Subido 49.9 MB", al bajar la DB del
bucket: 0 predicciones del 2026-05-24, último `run_id=855` (de 2026-05-21).

Investigación con md5 de descargas múltiples:

| Descarga | Cuando | md5 |
|---|---|---|
| post t2slg upload | después del primer backend exec | `29d4ac35...` |
| post harvester `:50` | después del harvester sobrescribió | `29d4ac35...` |
| post backend `4jbt4` (harvester pausado) | después de re-correr backend | `29d4ac35...` |

El backend SUBIÓ 49.9 MB pero la DB del bucket **no contiene sus escrituras**.
Esto es 100% reproducible. Cuando se pausó el harvester (descartando race)
seguía sin persistir.

**Causa:** la conexión SQLite mantiene página dirty en cache después del
`conn.commit()`. `commit()` en SQLite con journal_mode=DELETE escribe al
rollback journal y luego al main DB. Pero si el proceso lee el archivo
inmediatamente (`blob.upload_from_filename("/tmp/live_data.db")`) sin haber
cerrado la conexión, **el OS puede no haber flusheado las páginas dirty**.

En particular en Cloud Run, `/tmp` es tmpfs (RAM-backed). Las páginas dirty
se quedan en el page cache hasta que la conexión libere sus locks. El upload
lee los bytes del filesystem que NO incluyen los últimos commits.

**Fix** (`live_pull.py:362`):

```python
conn.commit()
# Close the connection so SQLite flushes cached pages to disk BEFORE the
# subsequent GCS upload in live_job.py reads /tmp/live_data.db.
conn.close()

print(f"\nDone. run_id={run_id}")
return 0
```

Tras este fix (execution `6nbwr`), bajamos la DB y:
- 52 predicciones del 2026-05-24 ✅
- `run_id=856 | started=2026-05-24T19:11:33 | flights_predicted=52` ✅

### 3.6 — Race condition descubierto (no causó el crash actual, pero va a)

Antes del fix de `conn.close()`, descartando la teoría del flush, indagamos
si el harvester estaba pisando las escrituras. Timeline observado:

```
18:46:10  backend (t2slg) start → download DB (con harvester id≤395)
18:50:30  backend uploads (49.9 MB)
18:50:31  harvester (jvfkm) start → download DB
18:50:35  backend complete
18:52:34  harvester uploads (49.9 MB, sin las predicciones del backend)
```

El harvester arrancó **1 segundo después** del upload del backend. Bajó la
versión "fresca" del backend (con las predicciones), pero al ejecutar su loop
de captura y RE-SUBIR, su SQLite no había materializado las predicciones del
backend (mismo bug que el §3.5 pero en el lado del harvester: tampoco
hace `conn.close()`).

**Mitigación temporal:** pausar `ontimeai-harvester-scheduler`. Hecho.

**Solución arquitectónica pendiente:** el patrón "download → modify → upload
del archivo entero" no es seguro con múltiples writers. Opciones evaluadas:

1. **Stagger schedules** (cron `:00,:30` backend + `:10,:25,:40,:55` harvester
   con margen >2 min entre executions). Frágil: si un job tarda más de lo
   normal, vuelve el race.

2. **Tabla `predictions` en bucket separado** — el backend solo lee `flights`
   del bucket compartido y escribe `predictions` aparte. Cambio de schema.

3. **Cloud SQL / Firestore** en lugar de SQLite — costo +$10-30/mes pero
   resuelve todo. Requiere refactor de `open_db()` y queries.

4. **Lock via GCS object** — el escritor crea `live_data.db.lock` blob antes
   de bajar/escribir/subir. Otros writers esperan o abortan. Hack.

Recomendación: opción 2 (bucket separado) o 3 (Cloud SQL) cuando se priorize
re-activar la operación continua. Por ahora, ambos schedulers PAUSED.

---

## 4. Validación end-to-end — las 52 predicciones reales

**Run 856 — 2026-05-24T19:11:33 UTC, model `4year_v9`, threshold quantile@0.22 = 0.5810**

### Distribución por riesgo

| Bucket | N | Avg proba |
|---|---|---|
| HIGH (≥0.70) | 7 | 0.805 |
| MED (0.50-0.70) | 7 | 0.556 |
| LOW (0.30-0.50) | 15 | 0.430 |
| VERY LOW (<0.30) | 23 | 0.181 |

### Distribución por dirección

| Flow | N | Avg proba | Predicted delayed |
|---|---|---|---|
| ARR_TO_ATL | 26 | 0.559 | 13 |
| DEP_FROM_ATL | 26 | 0.215 | 0 |

El modelo es **mucho más bajista para departures de ATL** (las controla la
operación de Delta directamente) y volátil para arrivals (dependen del origen).
Coincide con la distribución histórica de delays en KATL.

### Top 5 alto riesgo (validables ex-post cuando aterricen)

| Flight | Carrier | Route | Sched UTC | Proba | Lineage |
|---|---|---|---|---|---|
| SWA8509 | WN | ISP→ATL | 17:00 | 0.948 | no |
| DAL1137 | DL | ALB→ATL | 17:00 | 0.882 | yes |
| DAL1549 | DL | TLH→ATL | 18:43 | 0.817 | yes |
| FFT1669 | F9 | RDU→ATL | 16:54 | 0.797 | yes |
| DAL940 | DL | SAV→ATL | 17:59 | 0.756 | yes |

Todos arrivals, 4/5 con `inbound_fa_flight_id` resuelto via lineage. Ground
truth disponible cuando aterricen — el actual `arr_delay_min` se cargará en la
tabla `actuals` y el SQL `JOIN predictions × actuals` cerrará el ciclo.

### Top 5 bajo riesgo

| Flight | Carrier | Route | Sched UTC | Proba |
|---|---|---|---|---|
| DAL3095 | DL | ATL→GSP | 19:15 | 0.054 |
| EDV5128 | 9E | ATL→XNA | 19:20 | 0.054 |
| EDV5246 | 9E | ATL→ABE | 19:20 | 0.068 |
| DAL3146 | DL | ATL→MSY | 19:15 | 0.084 |
| DAL3133 | DL | ATL→GNV | 19:20 | 0.088 |

Todos departures de ATL hacia destinos regionales. Coherente.

---

## 5. Archivos modificados

| Archivo | Cambio | Línea |
|---|---|---|
| `ontimeai/live.py` | NA-safe en `np.where(ORIGIN.eq("ATL"))` | 632-637 |
| `ontimeai/model.py` | `params={"num_threads": 1}` en `lgb.Booster()` | 153-156 |
| `requirements-api.txt` | `lightgbm>=4.0` → `lightgbm==4.5.0` | 3 |
| `live_pull.py` | `DISABLE_LINEAGE_FALLBACK` env + `conn.close()` antes return | 251-263, 364 |
| `artifacts/4year_v9/model.lgb` | CRLF → LF (in-place) | — |
| `artifacts/4year_v9_recal/model.lgb` | CRLF → LF (in-place) | — |
| `artifacts/*/model.lgb.crlf.bak` | backups del CRLF original | (nuevos) |
| `.gitignore` | excluye `*.crlf.bak` | — |

**Nota sobre `lightgbm==4.5.0`:** intentamos este pin durante el diagnóstico
para descartar incompatibilidad de versión. Resultó irrelevante para el bug
(que era CRLF). Se mantiene porque pinear es mejor que `>=4.0` que abre la
puerta a cualquier major futura.

**Nota sobre `params={"num_threads": 1}`:** igual que arriba — irrelevante para
el bug pero buena higiene en Cloud Run con CPU limitada.

**Nota sobre `DISABLE_LINEAGE_FALLBACK` env var:** el fallback estaba bien todo
el tiempo. La env var queda como flag de diagnóstico por si reaparecen sospechas.

---

## 6. Recursos GCP que se modificaron

| Recurso | Cambio |
|---|---|
| `ontimeai-live-pull` (Cloud Run Job) | Imagen actualizada 3 veces. Última: `gcr.io/ontimeai/ontimeai-live-pull:latest` digest `sha256:0fd530...` |
| Env vars del job | Agregadas: `DISABLE_LINEAGE_FALLBACK=1`, `OMP_NUM_THREADS=1`, `LIGHTGBM_NUM_THREADS=1`. Mantenidas (ya estaban): `GCS_BUCKET`, `ACTIVE_MODEL=4year_v9`, `LIVE_DATA_SOURCE=aeroapi`, `AEROAPI_KEY` (secret) |
| `ontimeai-harvester-scheduler` | PAUSED desde 2026-05-24T18:55 UTC aprox |
| `ontimeai-pull-scheduler` | sigue PAUSED (estaba así antes de la sesión) |
| `gs://ontimeai-live-db/live_data.db` | Última escritura `2026-05-24T19:14:28Z`, contiene run 856 + 52 predicciones |
| Cloud Build executions | 5 builds nuevos (todos exitosos): `36deed8e`, `5236811f`, `4e602467`, `af097aa8`, `e932d012` |

---

## 7. Lo que está pendiente (orden de prioridad)

### 7.1 🔴 Race condition arquitectónico (BLOQUEANTE para automatización)
Decidir entre stagger / bucket separado / Cloud SQL. Sin esto NO se pueden
re-activar ambos schedulers concurrentes. Ver §3.6.

### 7.2 🟡 Reactivar `ontimeai-pull-scheduler`
Una vez resuelto §7.1:
```bash
gcloud scheduler jobs resume ontimeai-pull-scheduler --location=us-central1 --project=ontimeai
gcloud scheduler jobs resume ontimeai-harvester-scheduler --location=us-central1 --project=ontimeai
```

### 7.3 🟡 Validar las 52 predicciones contra actuals
A medida que los vuelos aterricen en las próximas 6h, comparar:
```sql
SELECT p.fa_flight_id, p.proba_delay, p.predicted_delay,
       a.arr_delay_min, (a.arr_delay_min > 15) AS truth
FROM predictions p
JOIN actuals a USING (fa_flight_id)
WHERE p.predicted_at_utc >= '2026-05-24T19:00:00' AND a.arr_delay_min IS NOT NULL;
```

Calcular Brier / AUC sobre esta muestra. Si AUC del run 856 ≥ 0.70, confirma
que el modelo predice bien tras la curación de datos del harvester.

### 7.4 🟢 Pre-commit hook para evitar CRLF en `*.lgb`
Para que no vuelva a entrar un model.lgb corrupto. Algo simple:
```bash
# .git/hooks/pre-commit
git diff --cached --name-only | grep -E '\.lgb$' | while read f; do
  if grep -q $'\r' "$f"; then
    echo "ERROR: $f has CRLF line endings — run: tr -d '\\r' < $f > $f.tmp && mv $f.tmp $f"
    exit 1
  fi
done
```

O agregar `*.lgb text eol=lf` a `.gitattributes`.

### 7.5 🟢 Actualizar `OnTimeAI-Scrapper/SESSION_HALLAZGOS.md`
Marcar `§4.4` como hipótesis incorrecta. El bug NO era LightGBM SIGSEGV en
predict — era CRLF en model.lgb durante load.

### 7.6 🟢 Cosmético — eliminar warning de `netCDF4`
Sigue apareciendo en logs ("v7 wind features failed"). Sin impacto operacional.

---

## 8. Lecciones aprendidas

1. **`SESSION_HALLAZGOS.md` documentaba el síntoma final, no la causa raíz.**
   El crash visible (cascada de "Model format error" + SIGSEGV) parecía
   confirmar que LightGBM moría en predict. En realidad moría en LOAD, pero el
   stack trace no quedaba claro porque el cascade ocultaba el origen.

2. **El bug "encadenado" engaña.** El bug NA (§3.1) ocultaba el bug CRLF (§3.4)
   ocultaba el bug del flush (§3.5). Cada fix destapaba el siguiente. Hubo
   que ejecutar 5 veces el job para destapar los 3.

3. **CRLF en archivos generados en Windows es un riesgo real.** Especialmente
   cuando esos archivos son consumidos por parsers C/C++ (LightGBM) en Linux.
   Defensa: `.gitattributes` con `* text=auto eol=lf` o equivalente.

4. **Race conditions en bucket compartido se enmascaran como otros bugs.** En
   este caso, el upload "se hacía" pero el contenido era el anterior. md5sum
   fue la única forma de detectar la inconsistencia objetivamente.

5. **`SESSION_HALLAZGOS.md §8.1` sugería bypass del fallback como diagnóstico.
   No tenía relación con el bug real**, pero el approach (env vars de bypass +
   single-thread) es razonable como técnica general.

---

## 9. Comandos para reproducir el debug

```bash
# 1. Auth
gcloud auth login santiagocarranzazinny@gmail.com
gcloud config set project ontimeai

# 2. Bajar DB actual
gsutil cp gs://ontimeai-live-db/live_data.db tmp/live_data.now.db

# 3. Inspeccionar bug NA (filas con origin NULL del harvester)
sqlite3 tmp/live_data.now.db "
SELECT COUNT(*) FROM flights f
JOIN actuals a ON a.stable_id = f.stable_id
WHERE f.stable_id IS NOT NULL AND (f.origin IS NULL OR f.dest IS NULL);
"

# 4. Verificar CRLF en model.lgb
file artifacts/4year_v9/model.lgb
od -c artifacts/4year_v9/model.lgb | head -3  # debe ser 'tree\nversion=v4\n' (sin \r)

# 5. Rebuild + execute backend
gcloud builds submit --config=cloudbuild-job.yaml --project=ontimeai
gcloud run jobs execute ontimeai-live-pull --region=us-central1 --project=ontimeai --wait

# 6. Verificar predicciones nuevas
gsutil cp gs://ontimeai-live-db/live_data.db tmp/live_data.verify.db
sqlite3 tmp/live_data.verify.db "SELECT COUNT(*) FROM predictions WHERE predicted_at_utc >= '2026-05-24';"
```

---

## 10. Resumen one-liner

> El backend estaba caído desde 2026-05-21 por un `model.lgb` con CRLF y un
> bug pandas-3 con datos del harvester. Tras 4 fixes en la sesión actual,
> generó 52 predicciones reales del run 856 (AUC pendiente de validación con
> actuals). El race condition entre backend y harvester sigue siendo bloqueante
> para automatizar.
