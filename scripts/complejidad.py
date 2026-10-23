#!/usr/bin/env python3
"""Gate y reporte de complejidad ciclomática del código productivo.

Dos herramientas, dos roles:

* ``ruff`` (regla C901, McCabe) es el **gate**: ninguna función del código
  productivo puede superar ``UMBRAL`` decisiones, salvo las que ya lo superaban
  cuando se instaló el gate y figuran en ``.complejidad-baseline.json``. Esa
  lista es un trinquete: una función exenta puede bajar o salir de la lista,
  pero no entrar una nueva ni crecer por encima del valor con el que entró.
* ``radon`` es el **reporte**: complejidad por bloque, promedio y distribución
  por rango, para el resumen del job y para seguir la evolución en el tiempo.

Uso::

    python scripts/complejidad.py --check                 # gate (CI y local)
    python scripts/complejidad.py --reporte salida.json   # radon por bloque
    python scripts/complejidad.py --resumen resumen.md    # markdown del reporte
    python scripts/complejidad.py --actualizar-baseline   # achica la lista exenta

Ver docs/COMPLEJIDAD.md para el porqué del umbral y del alcance.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# Sólo lo que corre en producción, definido por lo que copian las tres imágenes
# (Dockerfile, Dockerfile.job, Dockerfile.watchdog): el servicio HTTP, el job de
# inferencia, el vigía, el paquete `ontimeai` y `feature_engineering_v7`, que
# el runtime importa. Los scripts de análisis y de entrenamiento no se
# despliegan, así que su complejidad no es un riesgo operativo.
PRODUCTIVO = [
    "api.py",
    "predict.py",
    "live_pull.py",
    "live_job.py",
    "watchdog.py",
    "ontimeai",
    "feature_engineering_v7",
    "scripts/prune_db.py",
]

# McCabe clásico: hasta 10 es una función simple; 11-20 moderada; más de 20
# compleja y difícil de probar exhaustivamente.
UMBRAL = 10

BASELINE = RAIZ / ".complejidad-baseline.json"

_MENSAJE = re.compile(r"`(?P<nombre>[^`]+)` is too complex \((?P<cc>\d+) > \d+\)")


def _rutas_existentes() -> list[str]:
    return [p for p in PRODUCTIVO if (RAIZ / p).exists()]


def violaciones() -> dict[str, int]:
    """Funciones productivas por encima del umbral, ignorando los ``# noqa``.

    Se ignoran a propósito: el gate compara contra la baseline, no contra las
    marcas en el código. Si alguien agrega un ``# noqa: C901`` a una función
    nueva, acá igual aparece y el check la rechaza.
    """
    cmd = [
        sys.executable, "-m", "ruff", "check",
        "--select", "C901",
        "--config", f"lint.mccabe.max-complexity = {UMBRAL}",
        "--ignore-noqa",
        "--output-format", "json",
        "--exit-zero",
        *_rutas_existentes(),
    ]
    salida = subprocess.run(cmd, cwd=RAIZ, capture_output=True, text=True, check=False)
    if salida.returncode not in (0, 1):
        sys.stderr.write(salida.stderr)
        raise SystemExit(f"ruff terminó con código {salida.returncode}")
    resultado: dict[str, int] = {}
    for item in json.loads(salida.stdout or "[]"):
        m = _MENSAJE.search(item["message"])
        if not m:
            continue
        rel = Path(item["filename"]).resolve().relative_to(RAIZ).as_posix()
        resultado[f"{rel}::{m['nombre']}"] = int(m["cc"])
    return resultado


def cargar_baseline() -> dict[str, int]:
    if not BASELINE.exists():
        return {}
    datos = json.loads(BASELINE.read_text())
    if datos.get("umbral") != UMBRAL:
        raise SystemExit(
            f"la baseline fue tomada con umbral {datos.get('umbral')} y el script usa {UMBRAL}: "
            "regenerarla con --actualizar-baseline"
        )
    return {k: int(v) for k, v in datos.get("exentas", {}).items()}


def guardar_baseline(exentas: dict[str, int]) -> None:
    datos = {
        "_": "Funciones productivas por encima del umbral cuando se instaló el gate. "
             "Sólo puede achicarse: ver docs/COMPLEJIDAD.md.",
        "umbral": UMBRAL,
        "exentas": dict(sorted(exentas.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    BASELINE.write_text(json.dumps(datos, indent=2, ensure_ascii=False) + "\n")


def check() -> int:
    actuales = violaciones()
    exentas = cargar_baseline()

    nuevas = {k: v for k, v in actuales.items() if k not in exentas}
    crecieron = {k: (exentas[k], v) for k, v in actuales.items() if k in exentas and v > exentas[k]}
    resueltas = {k: v for k, v in exentas.items() if k not in actuales}

    print(f"umbral {UMBRAL} · {len(actuales)} funciones por encima · {len(exentas)} exentas por baseline")
    if resueltas:
        print("\nYa no superan el umbral (sacarlas de la baseline con --actualizar-baseline):")
        for k, v in sorted(resueltas.items()):
            print(f"  ✓ {k}  (entró con {v})")

    fallo = False
    if nuevas:
        fallo = True
        print(f"\n✗ {len(nuevas)} función(es) nueva(s) por encima de {UMBRAL}:")
        for k, v in sorted(nuevas.items(), key=lambda kv: -kv[1]):
            print(f"  {k}  CC={v}")
        print("  Dividirla hasta bajar de 10. La baseline no acepta entradas nuevas.")
    if crecieron:
        fallo = True
        print(f"\n✗ {len(crecieron)} función(es) exenta(s) que crecieron:")
        for k, (antes, ahora) in sorted(crecieron.items()):
            print(f"  {k}  {antes} → {ahora}")
        print("  Una función exenta puede bajar, no subir.")
    if not fallo:
        print("\n✓ sin deuda nueva de complejidad")
    return 1 if fallo else 0


def reporte_radon() -> dict:
    cmd = [sys.executable, "-m", "radon", "cc", "-j", *_rutas_existentes()]
    salida = subprocess.run(cmd, cwd=RAIZ, capture_output=True, text=True, check=True)
    crudo = json.loads(salida.stdout)
    bloques = []
    for archivo, items in crudo.items():
        if isinstance(items, dict):  # error de parseo de radon
            continue
        for b in items:
            nombre = f"{b['classname']}.{b['name']}" if b.get("classname") else b["name"]
            bloques.append({
                "archivo": Path(archivo).as_posix(),
                "nombre": nombre,
                "tipo": b["type"],
                "linea": b["lineno"],
                "cc": b["complexity"],
                "rango": b["rank"],
            })
    bloques.sort(key=lambda b: (-b["cc"], b["archivo"], b["linea"]))
    total = len(bloques)
    promedio = round(sum(b["cc"] for b in bloques) / total, 2) if total else 0.0
    return {
        "umbral": UMBRAL,
        "alcance": _rutas_existentes(),
        "bloques": total,
        "promedio": promedio,
        "por_rango": dict(sorted(Counter(b["rango"] for b in bloques).items())),
        "sobre_umbral": sum(1 for b in bloques if b["cc"] > UMBRAL),
        "sobre_20": sum(1 for b in bloques if b["cc"] > 20),
        "detalle": bloques,
    }


def resumen_markdown(rep: dict, exentas: dict[str, int]) -> str:
    lineas = [
        "### Complejidad ciclomática (código productivo)",
        "",
        f"| Bloques | Promedio | > {UMBRAL} (radon) | > 20 | Exentas en baseline (ruff) |",
        "|---|---|---|---|---|",
        f"| {rep['bloques']} | {rep['promedio']} | {rep['sobre_umbral']} | {rep['sobre_20']} | {len(exentas)} |",
        "",
        "Por rango: " + ", ".join(f"{r}={n}" for r, n in rep["por_rango"].items()),
        "",
        "| Función | CC | Archivo |",
        "|---|---|---|",
    ]
    for b in rep["detalle"][:10]:
        lineas.append(f"| `{b['nombre']}` | {b['cc']} | `{b['archivo']}:{b['linea']}` |")
    lineas.append("")
    return "\n".join(lineas)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="gate: falla ante deuda nueva")
    ap.add_argument("--reporte", type=Path, help="escribe el reporte de radon (JSON)")
    ap.add_argument("--resumen", type=Path, help="agrega el resumen en markdown a este archivo")
    ap.add_argument("--actualizar-baseline", action="store_true",
                    help="reescribe la baseline con las violaciones actuales (sólo puede achicarse)")
    args = ap.parse_args(argv)

    codigo = 0
    if args.actualizar_baseline:
        actuales = violaciones()
        previa = cargar_baseline() if BASELINE.exists() else None
        if previa is not None:
            nuevas = set(actuales) - set(previa)
            crecieron = [k for k in actuales if k in previa and actuales[k] > previa[k]]
            if nuevas or crecieron:
                print("la baseline sólo puede achicarse; primero bajar la complejidad de:")
                for k in sorted(nuevas | set(crecieron)):
                    print(f"  {k}  CC={actuales[k]}")
                return 1
        guardar_baseline(actuales)
        print(f"baseline escrita con {len(actuales)} exentas en {BASELINE.name}")
    if args.check:
        codigo = check()
    if args.reporte or args.resumen:
        rep = reporte_radon()
        if args.reporte:
            args.reporte.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
            print(f"reporte radon en {args.reporte} ({rep['bloques']} bloques, promedio {rep['promedio']})")
        if args.resumen:
            with args.resumen.open("a") as fh:
                fh.write(resumen_markdown(rep, cargar_baseline() if BASELINE.exists() else {}))
    if not (args.check or args.reporte or args.resumen or args.actualizar_baseline):
        ap.print_help()
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
