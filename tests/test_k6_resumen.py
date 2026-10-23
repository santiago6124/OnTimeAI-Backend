"""`scripts/k6_resumen.py`: la tabla que el workflow escribe en el step summary."""
from __future__ import annotations

import json
from pathlib import Path

from scripts import k6_resumen


def _summary(escenario: str, *, p95_ok: bool = True, errores: float = 0.0) -> dict:
    """Un `handleSummary` de k6 recortado a lo que lee el script."""
    return {
        "ontimeai": {"escenario": escenario},
        "metrics": {
            "http_req_duration": {
                "type": "trend",
                "values": {"med": 48.2, "p(95)": 120.7, "p(99)": 180.1, "avg": 55.0},
                "thresholds": {"p(95)<800": {"ok": p95_ok}},
            },
            "http_req_failed": {
                "type": "rate",
                "values": {"rate": errores, "passes": 3, "fails": 497},
                "thresholds": {"rate<0.01": {"ok": errores < 0.01}},
            },
            "checks": {"type": "rate", "values": {"rate": 1.0}},
            "http_reqs": {"type": "counter", "values": {"count": 500, "rate": 16.6}},
            "iterations": {"type": "counter", "values": {"count": 62, "rate": 2.1}},
            "vus_max": {"type": "gauge", "values": {"value": 50, "min": 50, "max": 50}},
        },
    }


def _escribir(tmp_path: Path, nombre: str, contenido: dict) -> Path:
    ruta = tmp_path / f"{nombre}-summary.json"
    ruta.write_text(json.dumps(contenido), encoding="utf-8")
    return ruta


def test_una_fila_por_escenario_con_los_numeros_clave(tmp_path: Path, capsys) -> None:
    rutas = [
        _escribir(tmp_path, "smoke", _summary("smoke")),
        _escribir(tmp_path, "stress", _summary("stress")),
    ]

    assert k6_resumen.main([str(r) for r in rutas]) == 0

    salida = capsys.readouterr().out
    assert salida.startswith("## Pruebas de carga con k6")
    assert "| smoke | 50 | 62 | 16.6 | 48 | 121 | 180 | 0.00 % | 100.00 % | OK (2/2) |" in salida
    assert salida.count("| stress |") == 1


def test_un_umbral_fallido_se_nombra_y_el_codigo_de_salida_lo_dice(tmp_path: Path, capsys) -> None:
    ruta = _escribir(tmp_path, "soak", _summary("soak", p95_ok=False, errores=0.03))

    assert k6_resumen.main([str(ruta)]) == 2

    salida = capsys.readouterr().out
    assert "FALLA http_req_duration p(95)<800; http_req_failed rate<0.01" in salida
    assert "3.00 %" in salida


def test_agrega_al_archivo_de_salida_y_avisa_los_que_faltan(tmp_path: Path) -> None:
    ruta = _escribir(tmp_path, "smoke", _summary("smoke"))
    salida = tmp_path / "step-summary.md"
    salida.write_text("# Antes\n", encoding="utf-8")

    codigo = k6_resumen.main([
        str(ruta), str(tmp_path / "stress-summary.json"), "--salida", str(salida),
        "--titulo", "## k6",
    ])

    assert codigo == 0
    texto = salida.read_text(encoding="utf-8")
    assert texto.startswith("# Antes\n## k6\n")
    assert "Sin resultado" in texto and "stress-summary.json" in texto


def test_sin_ningun_summary_falla_con_mensaje(tmp_path: Path, capsys) -> None:
    assert k6_resumen.main([str(tmp_path / "nada.json")]) == 1
    assert "No hay ningún summary" in capsys.readouterr().err


def test_tolera_metricas_ausentes(tmp_path: Path) -> None:
    """Un summary sin umbrales ni percentiles raros (p. ej. un smoke abortado)."""
    recortado = {"metrics": {"http_reqs": {"values": {"count": 3}}}}
    fila = k6_resumen.fila(recortado, "smoke")
    assert fila["Escenario"] == "smoke"
    assert fila["p95 ms"] == "—" and fila["req/s"] == "—"
    assert fila["Umbrales"] == "sin umbrales"
