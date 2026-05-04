# Cron / launchd setup (macOS)

Pensado para correr en tu laptop personal mientras el backend no tenga server.
Usa `launchd` (lo que reemplaza a `cron` en macOS) con dos LaunchAgents:

- **`com.ontimeai.tick`** — cada 2h, llama `live_pull.py` con flags conservadores
- **`com.ontimeai.metrics`** — diario a las 04:00 local, computa F1/AUC sobre las predicciones evaluadas

## Instalar

```bash
cd /Users/lologalaverna/Projects/Tesis/OnTimeAI-Backend
chmod +x scripts/*.sh
./scripts/install.sh
```

Eso copia los plists a `~/Library/LaunchAgents/` y los activa. No requiere sudo.

## Desinstalar / pausar

```bash
# Pausa temporal (sigue cargado pero el wrapper sale en 0):
touch .cron_disabled

# Reactivar:
rm .cron_disabled

# Desinstalar permanente:
./scripts/uninstall.sh
```

## Tunear flags sin tocar código

El wrapper `cron_tick.sh` lee variables de entorno. Para ser más/menos agresivo,
editá los valores hardcodeados o exportalos en `~/.zshenv`:

| Variable | Default | Efecto |
|---|---|---|
| `ONTIMEAI_SCHEDULE_HOURS` | 2 | Ventana de schedule pull |
| `ONTIMEAI_MAX_PAGES` | 2 | Páginas cursor por endpoint |
| `ONTIMEAI_CHAIN_WALK_MAX` | 5 | Máx calls de chain-walk por tick |
| `ONTIMEAI_TARGET_POS_RATE` | 0.26 | Target pos_pred_rate del quantile threshold |
| `ONTIMEAI_MIN_TICK_DELTA` | 5400 | Min segundos entre ticks (anti catch-up) |

## Costo estimado

| Componente | Calls/tick | $/tick |
|---|---|---|
| `scheduled_departures` (max 2 pages) | 2 | 0.010 |
| `arrivals` actuals (max 2 pages) | 2 | 0.010 |
| chain-walk (cap 5) | 5 | 0.025 |
| IEM weather | 0 (gratis) | 0 |
| **Total tick** | **9** | **$0.045** |

UTC active hours = `13–03` (cubre ATL ops 09 EDT — 23 EDT).
Con `StartInterval=7200` (2h) → fires ~12/día, de los cuales **6 caen en active hours**.

**Costo diario: 6 × $0.045 = $0.27 USD**
**5+ días sostenibles con $1.50 USD restantes**

`com.ontimeai.metrics` es 100% gratis (lee solo la SQLite local).

## Salvaguardas operacionales

El wrapper `cron_tick.sh` tiene 4 guard rails contra over-spending:

1. **Kill-switch**: archivo `.cron_disabled` corta la ejecución antes de cualquier API call.
2. **UTC gate**: si la hora actual UTC ∈ `[4, 12]` (ATL madrugada), tick es no-op.
3. **Min interval**: si `now - last_successful_tick < ONTIMEAI_MIN_TICK_DELTA` (90min default), tick es no-op. Evita que tras unas horas con la laptop dormida, launchd lance 4 ticks back-to-back al despertar.
4. **PID lock**: si hay otro tick corriendo, abortar.

## Logs

- `logs/cron_YYYYMMDD.log` — output del wrapper + del `live_pull.py`
- `logs/cron_metrics_YYYYMMDD.log` — output del nightly
- `logs/launchd_tick.{out,err}` — stdout/stderr del proceso launchd (para diagnosticar fallas de bootstrap)

Para ver el último tick:

```bash
tail -F logs/cron_$(date -u +%Y%m%d).log
```

Para ver estado del agente:

```bash
launchctl print "gui/$(id -u)/com.ontimeai.tick" | head -20
```

## Disparar un tick manual (sin esperar al próximo fire de launchd)

```bash
bash scripts/cron_tick.sh
```

Pasa por todas las salvaguardas (kill-switch, UTC gate, min interval). Para forzar:

```bash
ONTIMEAI_MIN_TICK_DELTA=0 bash scripts/cron_tick.sh
```

## Comportamiento con la laptop cerrada

`launchd` LaunchAgents **NO despiertan** la Mac. Comportamiento real:

- Laptop abierta: tick cada 2h normalmente
- Laptop cerrada/dormida: los ticks que tocaba se pierden — al despertar, launchd dispara **un solo** tick (el guard `min interval` evita catch-up storms)
- Si querés que la Mac se despierte para correr (consume batería): se puede con `pmset` + sudo, no recomendado para el MVP

## Inspeccionar la SQLite live

```bash
sqlite3 -header -column live_data.db "
SELECT
  COUNT(*) as predictions_total,
  SUM(CASE WHEN threshold_strategy LIKE 'quantile%' THEN 1 ELSE 0 END) as with_quantile,
  AVG(proba_delay) as mean_proba,
  AVG(predicted_delay) as pos_pred_rate
FROM predictions;
"
```

## Reset completo (si algo se desbarata)

```bash
./scripts/uninstall.sh
rm -f live_data.db .last_tick_utc .cron_tick.lock .cron_disabled
rm -rf logs
./scripts/install.sh
```
