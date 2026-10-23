#!/usr/bin/env python3
"""Resume los `*-summary.json` de k6 en una tabla Markdown.

Una fila por escenario: VUs, iteraciones, req/s, p50/p95/p99 de
`http_req_duration`, tasa de error HTTP y el resultado de los umbrales. Es lo
que el workflow `stress.yml` escribe en `$GITHUB_STEP_SUMMARY`, y lo que se
copia a `docs/STRESS_K6.md` después de una PoC.

    python scripts/k6_resumen.py k6/resultados/*-summary.json
    python scripts/k6_resumen.py k6/resultados/*.json --salida "$GITHUB_STEP_SUMMARY"

Los JSON los escribe `handleSummary` de cada script de `k6/` (ver `k6/lib.js`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

COLUMNAS = (
    "Escenario", "VUs máx", "Iteraciones", "req/s", "p50 ms", "p95 ms",
    "p99 ms", "Errores HTTP", "Checks", "Umbrales",
)


def _valores(metricas: dict[str, Any], nombre: str) -> dict[str, Any]:
    return (metricas.get(nombre) or {}).get("values") or {}


def _ms(valor: Any) -> str:
    return "—" if valor is None else f"{float(valor):.0f}"


def _porcentaje(valor: Any) -> str:
    return "—" if valor is None else f"{100 * float(valor):.2f} %"


def umbrales(metricas: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Los umbrales declarados, separados en los que pasaron y los que no.

    k6 cuelga cada umbral de su métrica: `metrics[m].thresholds[expr].ok`.
    """
    ok, fallidos = [], []
    for nombre, metrica in metricas.items():
        for expresion, resultado in ((metrica or {}).get("thresholds") or {}).items():
            etiqueta = f"{nombre} {expresion}"
            (ok if (resultado or {}).get("ok") else fallidos).append(etiqueta)
    return ok, fallidos


def fila(resumen: dict[str, Any], nombre: str) -> dict[str, str]:
    metricas = resumen.get("metrics") or {}
    duracion = _valores(metricas, "http_req_duration")
    vus = _valores(metricas, "vus_max")
    fallos = _valores(metricas, "http_req_failed")
    checks = _valores(metricas, "checks")
    ok, fallidos = umbrales(metricas)

    if fallidos:
        veredicto = "FALLA " + "; ".join(fallidos)
    elif ok:
        veredicto = f"OK ({len(ok)}/{len(ok)})"
    else:
        veredicto = "sin umbrales"

    escenario = (resumen.get("ontimeai") or {}).get("escenario") or nombre
    return {
        "Escenario": escenario,
        "VUs máx": _ms(vus.get("max", vus.get("value"))),
        "Iteraciones": _ms(_valores(metricas, "iterations").get("count")),
        "req/s": "—" if _valores(metricas, "http_reqs").get("rate") is None
                 else f"{_valores(metricas, 'http_reqs')['rate']:.1f}",
        "p50 ms": _ms(duracion.get("med")),
        "p95 ms": _ms(duracion.get("p(95)")),
        "p99 ms": _ms(duracion.get("p(99)")),
        # k6 informa las tasas como `rate`; versiones viejas usaban `value`.
        "Errores HTTP": _porcentaje(fallos.get("rate", fallos.get("value"))),
        "Checks": _porcentaje(checks.get("rate", checks.get("value"))),
        "Umbrales": veredicto,
    }


def tabla_markdown(filas: list[dict[str, str]]) -> str:
    lineas = [
        "| " + " | ".join(COLUMNAS) + " |",
        "|" + "|".join("---" for _ in COLUMNAS) + "|",
    ]
    lineas += ["| " + " | ".join(f[c] for c in COLUMNAS) + " |" for f in filas]
    return "\n".join(lineas) + "\n"


def leer(ruta: Path) -> dict[str, str]:
    with ruta.open(encoding="utf-8") as fh:
        resumen = json.load(fh)
    nombre = ruta.name.removesuffix("-summary.json")
    return fila(resumen, nombre)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("resumenes", nargs="+", type=Path,
                        help="archivos *-summary.json que escribe k6")
    parser.add_argument("--salida", type=Path, default=None,
                        help="archivo al que AGREGAR la tabla (p. ej. $GITHUB_STEP_SUMMARY); "
                             "sin esto se imprime por stdout")
    parser.add_argument("--titulo", default="## Pruebas de carga con k6")
    args = parser.parse_args(argv)

    existentes = [r for r in args.resumenes if r.is_file()]
    faltantes = [r for r in args.resumenes if not r.is_file()]
    if not existentes:
        print("No hay ningún summary de k6 para resumir.", file=sys.stderr)
        return 1

    texto = f"{args.titulo}\n\n{tabla_markdown([leer(r) for r in existentes])}"
    if faltantes:
        texto += "\nSin resultado (el paso no corrió o k6 no llegó a escribir el summary): " \
                 + ", ".join(r.name for r in faltantes) + "\n"

    if args.salida:
        with args.salida.open("a", encoding="utf-8") as fh:
            fh.write(texto)
    else:
        print(texto, end="")

    filas_fallidas = [leer(r) for r in existentes if leer(r)["Umbrales"].startswith("FALLA")]
    return 2 if filas_fallidas else 0


if __name__ == "__main__":
    sys.exit(main())
