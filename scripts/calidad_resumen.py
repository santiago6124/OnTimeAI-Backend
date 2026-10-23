#!/usr/bin/env python3
"""Resumen de calidad de un run de CI: cobertura, duplicación y código nuevo.

Lee lo que exista de ``coverage.xml`` (formato Cobertura de pytest-cov),
``jscpd/jscpd-report.json`` y ``diff-cover.md``, y agrega una tabla en markdown
al archivo indicado (en CI, ``$GITHUB_STEP_SUMMARY``). No falla si falta un
insumo: el gate lo ponen los pasos que los generan; esto sólo informa.

Uso::

    python scripts/calidad_resumen.py --cobertura coverage.xml \
        --jscpd jscpd/jscpd-report.json --diff-cover diff-cover.md \
        --salida resumen.md
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def leer_cobertura(ruta: Path) -> dict | None:
    """Totales de líneas y ramas del XML de Cobertura, más los archivos peor cubiertos."""
    if not ruta.exists():
        return None
    raiz = ET.parse(ruta).getroot()
    lineas = int(raiz.get("lines-valid", 0))
    cubiertas = int(raiz.get("lines-covered", 0))
    ramas = int(raiz.get("branches-valid", 0))
    ramas_cub = int(raiz.get("branches-covered", 0))
    archivos = []
    for cls in raiz.iter("class"):
        n = len(list(cls.iter("line")))
        if n >= 50:  # los archivos chicos no dicen nada
            archivos.append((round(float(cls.get("line-rate", 0)) * 100, 1), cls.get("filename"), n))
    archivos.sort()
    return {
        "lineas_pct": round(cubiertas / lineas * 100, 1) if lineas else 0.0,
        "lineas": lineas,
        "lineas_cubiertas": cubiertas,
        "ramas_pct": round(ramas_cub / ramas * 100, 1) if ramas else 0.0,
        "ramas": ramas,
        "ramas_cubiertas": ramas_cub,
        "peor_cubiertos": archivos[:5],
    }


def leer_duplicacion(ruta: Path) -> dict | None:
    if not ruta.exists():
        return None
    t = json.loads(ruta.read_text())["statistics"]["total"]
    return {
        "pct_lineas": t.get("percentage", 0.0),
        "lineas_duplicadas": t.get("duplicatedLines", 0),
        "lineas": t.get("lines", 0),
        "clones": t.get("clones", 0),
        "archivos": t.get("sources", 0),
    }


def markdown(cob: dict | None, dup: dict | None, diff_md: str | None) -> str:
    L = ["### Calidad del run", ""]
    L.append("| Cobertura de líneas | Cobertura de ramas | Duplicación (jscpd) |")
    L.append("|---|---|---|")
    c = f"{cob['lineas_pct']} % ({cob['lineas_cubiertas']}/{cob['lineas']})" if cob else "sin reporte"
    r = f"{cob['ramas_pct']} % ({cob['ramas_cubiertas']}/{cob['ramas']})" if cob else "sin reporte"
    d = f"{dup['pct_lineas']} % ({dup['lineas_duplicadas']}/{dup['lineas']} líneas, {dup['clones']} clones)" if dup else "sin reporte"
    L.append(f"| {c} | {r} | {d} |")
    if cob and cob["peor_cubiertos"]:
        L += ["", "Archivos grandes peor cubiertos:", ""]
        for pct, nombre, n in cob["peor_cubiertos"]:
            L.append(f"- `{nombre}`: {pct} % de {n} líneas")
    if diff_md:
        L += ["", "#### Cobertura sobre las líneas cambiadas", "", diff_md.strip()]
    L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cobertura", type=Path, default=Path("coverage.xml"))
    ap.add_argument("--jscpd", type=Path, default=Path("jscpd/jscpd-report.json"))
    ap.add_argument("--diff-cover", type=Path, default=Path("diff-cover.md"))
    ap.add_argument("--salida", type=Path, required=True, help="archivo al que se agrega el resumen")
    args = ap.parse_args(argv)

    cob = leer_cobertura(args.cobertura)
    dup = leer_duplicacion(args.jscpd)
    diff_md = args.diff_cover.read_text() if args.diff_cover.exists() else None
    texto = markdown(cob, dup, diff_md)
    with args.salida.open("a") as fh:
        fh.write(texto)
    print(texto)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
