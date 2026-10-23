#!/usr/bin/env python3
"""Arma la base fixture contra la que corren las pruebas de carga con k6.

Copia el `live_data.db` versionado a `--destino` y siembra `--vuelos` vuelos
de hoy —con predicción, SHAP, resultado para los que ya aterrizaron y una
pierna siguiente para la cascada— con `ontimeai.semillas`. El snapshot del
repo no tiene vuelos de hoy, así que sin la siembra `/flights` devuelve una
lista vacía y la prueba no ejercita nada.

    python scripts/make_fixture_db.py --destino /tmp/fixture.db --vuelos 200

Después se levanta la API apuntando a esa copia:

    DB_PATH_OVERRIDE=/tmp/fixture.db USERS_DB_PATH_OVERRIDE=/tmp/users.db \\
        uvicorn api:app --port 8000

Nunca se corre contra producción. Ver docs/STRESS_K6.md.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from ontimeai.semillas import Siembra, sembrar_vuelos_de_hoy  # noqa: E402

BASE_BUNDLEADA = RAIZ / "live_data.db"


def armar_fixture(destino: Path, vuelos: int, semilla: int,
                  origen: Path = BASE_BUNDLEADA) -> Siembra:
    """Copia `origen` en `destino` y siembra. Devuelve cuánto sembró."""
    if not origen.is_file():
        raise FileNotFoundError(f"no existe la base de origen {origen}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origen, destino)
    # Una copia hecha mientras la base estaba en WAL arrastraría los sidecars
    # de otro archivo; la fixture arranca limpia.
    for sufijo in ("-wal", "-shm"):
        Path(f"{destino}{sufijo}").unlink(missing_ok=True)
    return sembrar_vuelos_de_hoy(destino, n=vuelos, semilla=semilla)


def _entero_positivo(valor: str) -> int:
    n = int(valor)
    if n < 1:
        raise argparse.ArgumentTypeError("tiene que ser al menos 1")
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--destino", type=Path, required=True,
                        help="ruta de la base fixture a crear (se sobreescribe)")
    parser.add_argument("--vuelos", type=_entero_positivo, default=200,
                        help="vuelos de hoy a sembrar (default 200)")
    parser.add_argument("--semilla", type=int, default=2026,
                        help="semilla del generador, para reproducir la misma base")
    parser.add_argument("--origen", type=Path, default=BASE_BUNDLEADA,
                        help="base a copiar (default: el live_data.db del repo)")
    args = parser.parse_args(argv)

    siembra = armar_fixture(args.destino, args.vuelos, args.semilla, args.origen)
    mb = args.destino.stat().st_size / 1e6
    print(
        f"Base fixture en {args.destino} ({mb:.0f} MB): {siembra.vuelos} vuelos "
        f"de hoy sembrados — {siembra.predicciones} predicciones, {siembra.shap} "
        f"filas SHAP, {siembra.resultados} resultados, "
        f"{siembra.piernas_siguientes} piernas siguientes (semilla {args.semilla})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
