"""Compara estrategias de recalibrado FUERA de muestra, caminando hacia adelante.

`recalibrate_live.py` ajusta sobre todo el histórico y reporta las métricas
sobre los mismos datos con que ajustó. Eso siempre mejora: un calibrador que
vio la respuesta no puede empeorar el Brier del set que vio. Fue así como el
artefacto `4year_v9_recal` pasó los controles y después, en producción, movía
la probabilidad media a 4,5% contra una tasa real de 12,7%.

Acá cada estrategia se juzga contra vuelos que salieron DESPUÉS del corte, y
usando solo los resultados que ya estaban asentados en el corte —con un día de
margen, porque un vuelo que acaba de aterrizar todavía no tiene su actual
cargado—.

Uso:
    python3 evaluar_recalibrado.py --db /tmp/fresca.db
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from ontimeai.validacion_calibrado import MARGEN_ASENTAMIENTO, caminar_cortes

def cargar(db: str, umbral: float) -> pd.DataFrame:
    import sqlite3

    con = sqlite3.connect(f"file:{db}?immutable=1", uri=True)
    df = pd.read_sql_query(
        """
        SELECT p.fa_flight_id, p.proba_delay, p.predicted_at_utc,
               f.scheduled_out_utc, a.arr_delay_min
          FROM predictions p
          JOIN actuals a ON a.fa_flight_id = p.fa_flight_id
          JOIN flights  f ON f.fa_flight_id = p.fa_flight_id
         WHERE a.cancelled = 0 AND a.diverted = 0
           AND a.arr_delay_min IS NOT NULL
           AND f.scheduled_out_utc IS NOT NULL
           AND p.predicted_at_utc <= f.scheduled_out_utc
        """,
        con,
    )
    # La última predicción anterior a la salida: la que operaciones tiene en la
    # mano al decidir. Ordenar por tiempo, no por probabilidad (ver el
    # comentario largo en recalibrate_live.py).
    df = df.sort_values("predicted_at_utc").groupby("fa_flight_id", as_index=False).last()
    df["salida"] = pd.to_datetime(df.scheduled_out_utc, format="ISO8601", utc=True)
    df["y"] = (df.arr_delay_min > umbral).astype(int)
    df["p"] = df.proba_delay.astype(float)
    return df.sort_values("salida").reset_index(drop=True)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Error de calibración esperado: cuánto se aparta la promesa del resultado."""
    borde = np.linspace(0, 1, bins + 1)
    total = len(p)
    if total == 0:
        return float("nan")
    acc = 0.0
    for lo, hi in zip(borde[:-1], borde[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            acc += (m.sum() / total) * abs(p[m].mean() - y[m].mean())
    return float(acc)


def metricas(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "n": len(y),
        "brier": brier_score_loss(y, p) if len(y) else float("nan"),
        "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
        "ece": ece(y, p),
        "media": float(p.mean()) if len(p) else float("nan"),
        "real": float(y.mean()) if len(y) else float("nan"),
    }


def estrategias(dias_ventana: tuple[int, ...]) -> dict:
    """Nombre -> (metodo, dias de ventana o None para todo el historico)."""
    e = {"cruda": (None, None), "platt_todo": ("sigmoid", None), "iso_todo": ("isotonic", None)}
    for d in dias_ventana:
        e[f"platt_{d}d"] = ("sigmoid", d)
        e[f"iso_{d}d"] = ("isotonic", d)
    return e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--umbral", type=float, default=15.0)
    ap.add_argument("--horizonte", type=int, default=2,
                    help="Días de vuelos que se evalúan después de cada corte")
    ap.add_argument("--min-ajuste", type=int, default=400,
                    help="Mínimo de vuelos asentados para intentar ajustar")
    ap.add_argument("--ventanas", default="3,5,7")
    ap.add_argument("--por-corte", action="store_true",
                    help="Brier corte por corte: un promedio puede esconder que "
                         "una estrategia gana mucho un día y pierde el resto")
    args = ap.parse_args()

    vent = tuple(int(x) for x in args.ventanas.split(",") if x.strip())
    df = cargar(args.db, args.umbral)
    print(f"vuelos con resultado: {len(df):,}")
    print(f"rango de salidas: {df.salida.min():%Y-%m-%d} a {df.salida.max():%Y-%m-%d}")
    print(f"tasa real global: {df.y.mean():.1%}   probabilidad media: {df.p.mean():.1%}")

    estr = estrategias(vent)
    acum: dict[str, list] = {k: [] for k in estr}
    cortes = 0

    # Todas las estrategias caminan los MISMOS cortes. Sin esto, cada ventana
    # arranca donde le alcanza y acaban midiendo sobre períodos distintos:
    # medido el 11/10, iso_7d parecía ganar 20,7% porque le tocaban días
    # calmos, contra una cruda calculada sobre el doble de vuelos.
    desde = df.salida.min().normalize() + timedelta(days=min(vent) + 1)
    print()
    print(f"=== caminando cortes desde {desde:%d/%m}, "
          f"evaluando {args.horizonte} día(s) por corte ===")
    for nombre, (metodo, dias) in estr.items():
        if metodo is None:
            # La cruda se arma a partir de los tramos de prueba de cualquier
            # estrategia: son los mismos vuelos.
            continue
        cs = caminar_cortes(df["scheduled_out_utc"], df.y.values, df.p.values,
                            metodo=metodo, horizonte_dias=args.horizonte,
                            ventana_dias=dias, min_ajuste=args.min_ajuste,
                            desde=desde)
        acum[nombre] = [(c.y, c.p_recal, c.n_ajuste) for c in cs]
        if cs and not acum["cruda"]:
            acum["cruda"] = [(c.y, c.p_cruda, 0) for c in cs]
            cortes = len(cs)

    print(f"cortes evaluados: {cortes}")
    if not cortes:
        print("sin cortes suficientes; hace falta más historia")
        return 1

    print()
    print(f"   {'estrategia':<14}{'n test':>8}{'Brier':>9}{'AUC':>8}{'ECE':>8}"
          f"{'media':>8}{'real':>7}{'n ajuste':>10}")
    base = None
    filas = []
    for nombre in estr:
        if not acum[nombre]:
            continue
        y = np.concatenate([a for a, _, _ in acum[nombre]])
        p = np.concatenate([b for _, b, _ in acum[nombre]])
        nadj = int(np.mean([c for _, _, c in acum[nombre]]))
        m = metricas(y, p)
        if nombre == "cruda":
            base = m["brier"]
        filas.append((nombre, m, nadj))
    for nombre, m, nadj in filas:
        delta = ""
        if base and nombre != "cruda":
            d = (m["brier"] - base) / base * 100
            delta = f"  {d:+.1f}% Brier" + ("  <-- mejor" if d < -1 else
                                             ("  (peor)" if d > 1 else ""))
        print(f"   {nombre:<14}{m['n']:>8}{m['brier']:>9.4f}{m['auc']:>8.4f}{m['ece']:>8.4f}"
              f"{m['media']:>8.1%}{m['real']:>7.1%}{nadj if nadj else '-':>10}{delta}")

    if args.por_corte:
        print()
        print("=== Brier corte por corte (un promedio puede esconder la varianza) ===")
        cols = [n for n in ("cruda", "iso_todo", "iso_7d", "platt_todo") if acum[n]]
        print("   " + "".join(f"{c:>12}" for c in ["corte"] + cols))
        gana = {c: 0 for c in cols if c != "cruda"}
        for i in range(len(acum["cruda"])):
            fila = f"   {i+1:>12}"
            bs = {}
            for c in cols:
                y, p, _ = acum[c][i]
                bs[c] = brier_score_loss(y, p)
                fila += f"{bs[c]:>12.4f}"
            for c in gana:
                if bs[c] < bs["cruda"]:
                    gana[c] += 1
            print(fila)
        print()
        for c, v in gana.items():
            print(f"   {c:<12} le gana a la cruda en {v} de {len(acum['cruda'])} cortes")

    # La tabla por tramos de la ganadora frente a la cruda: es lo que ve el usuario.
    mejores = [f for f in filas if f[0] != "cruda"]
    if mejores:
        gan = min(mejores, key=lambda f: f[1]["brier"])[0]
        print()
        print(f"=== tramos: cruda vs {gan} ===")
        print(f"   {'promete':>12}{'n':>8}{'pasa':>8}      {'promete':>12}{'n':>8}{'pasa':>8}")
        for nombre in ("cruda", gan):
            pass
        yc = np.concatenate([a for a, _, _ in acum["cruda"]])
        pc = np.concatenate([b for _, b, _ in acum["cruda"]])
        yg = np.concatenate([a for a, _, _ in acum[gan]])
        pg = np.concatenate([b for _, b, _ in acum[gan]])
        for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 35), (35, 101)):
            mc = (pc * 100 >= lo) & (pc * 100 < hi)
            mg = (pg * 100 >= lo) & (pg * 100 < hi)
            iz = f"{lo:>4}-{hi:<4}%{mc.sum():>8}{yc[mc].mean():>8.1%}" if mc.sum() >= 30 else " " * 28
            de = f"{lo:>4}-{hi:<4}%{mg.sum():>8}{yg[mg].mean():>8.1%}" if mg.sum() >= 30 else ""
            print(f"   {iz}      {de}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
