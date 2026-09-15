"""
El calibrador post-cadena: aprende la traduccion y se apaga si no puede.

La cadena de ajustes modifica la probabilidad con constantes puestas a mano, asi
que su salida deja de ser una probabilidad: mostraba 89,5% para vuelos cuya tasa
real era 39%. El calibrador aprende esa traduccion de los vuelos que ya
aterrizaron.

Lo que estos tests protegen no es tanto que calibre bien —eso lo dice la
medicion— sino las tres guardas que impiden que haga dano:

  1. no se ajusta con muestra insuficiente
  2. no se aplica si esta vencido
  3. no se realimenta: aprende de la salida cruda, no de la ya calibrada

La tercera es la sutil. La primera y la segunda son la leccion de #15, donde un
artefacto congelado de agosto EMPEORABA las cosas dos meses despues.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from ontimeai import chain_calibration as cc


@pytest.fixture
def con() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.executescript(
        """
        CREATE TABLE predictions (
            fa_flight_id TEXT, stable_id TEXT, predicted_at_utc TEXT,
            proba_delay REAL, proba_chain REAL, prediction_phase TEXT
        );
        CREATE TABLE actuals (
            fa_flight_id TEXT, stable_id TEXT, arr_delay_min REAL,
            cancelled INTEGER, diverted INTEGER, settled_at_utc TEXT
        );
        """
    )
    cc.ensure_schema(c)
    return c


def _sembrar(con, n, *, exagera=2.5, demorados=0.3, hace_dias=1, fase="PRE_DEPARTURE"):
    """Vuelos donde la cadena exagera por un factor conocido.

    Si la tasa real es `p`, la cadena dice `min(p * exagera, 0.99)`. El
    calibrador tiene que aprender a deshacer eso.
    """
    rng = np.random.default_rng(20260915)
    cuando = (datetime.now(timezone.utc) - timedelta(days=hace_dias)).isoformat()
    for i in range(n):
        real = float(rng.uniform(0.02, 0.35))
        cadena = min(real * exagera, 0.99)
        y = 1 if rng.random() < real else 0
        fid = f"F{i:05d}"
        con.execute(
            "INSERT INTO predictions VALUES (?,?,?,?,?,?)",
            (fid, fid, cuando, cadena, cadena, fase),
        )
        con.execute(
            "INSERT INTO actuals VALUES (?,?,?,?,?,?)",
            (fid, fid, 40.0 if y else 2.0, 0, 0, cuando),
        )
    con.commit()


class TestGuardas:
    def test_no_se_ajusta_con_muestra_insuficiente(self, con) -> None:
        """Un isotonico con pocas muestras sigue el ruido e inventa una traduccion."""
        _sembrar(con, 100)
        assert cc.fit_chain_calibrator(con, min_samples=500) is None
        assert cc.load_chain_calibrator(con) is None

    def test_no_se_ajusta_si_no_hay_dos_clases(self, con) -> None:
        cuando = datetime.now(timezone.utc).isoformat()
        for i in range(600):
            fid = f"F{i:05d}"
            con.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?)",
                        (fid, fid, cuando, 0.5, 0.5, "PRE_DEPARTURE"))
            con.execute("INSERT INTO actuals VALUES (?,?,?,?,?,?)",
                        (fid, fid, 2.0, 0, 0, cuando))  # ninguno demorado
        con.commit()
        assert cc.fit_chain_calibrator(con, min_samples=100) is None

    def test_uno_vencido_no_se_aplica(self, con) -> None:
        """La leccion de #15: un corrector viejo corrige en la direccion equivocada."""
        _sembrar(con, 800)
        assert cc.fit_chain_calibrator(con, min_samples=100) is not None
        assert cc.load_chain_calibrator(con, max_age_days=7) is not None

        viejo = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        con.execute("UPDATE chain_calibrator SET fitted_at_utc = ?", (viejo,))
        con.commit()
        assert cc.load_chain_calibrator(con, max_age_days=7) is None, (
            "un calibrador de hace 30 dias no puede aplicarse"
        )

    def test_una_base_sin_la_tabla_no_rompe(self) -> None:
        c = sqlite3.connect(":memory:")
        assert cc.load_chain_calibrator(c) is None
        assert cc.calibrator_status(c)["present"] is False

    def test_una_base_sin_proba_chain_no_rompe(self) -> None:
        c = sqlite3.connect(":memory:")
        c.execute("CREATE TABLE predictions (fa_flight_id TEXT)")
        c.execute("CREATE TABLE actuals (fa_flight_id TEXT)")
        cc.ensure_schema(c)
        assert cc.fit_chain_calibrator(c) is None


class TestNoSeRealimenta:
    def test_aprende_de_la_salida_cruda_y_no_de_la_servida(self, con) -> None:
        """
        `proba_chain` es la salida de la cadena; `proba_delay` es lo que se
        sirve, ya calibrado. Ajustar sobre lo servido haria que cada reajuste
        aprendiera sobre valores ya corregidos y la correccion se aplicara dos
        veces, tres, n veces.
        """
        _sembrar(con, 800)
        # Se simula un ciclo ya calibrado: proba_delay bajo, proba_chain intacto.
        con.execute("UPDATE predictions SET proba_delay = proba_chain / 3.0")
        con.commit()

        filas = cc._training_rows(con, window_days=30)
        chain = {round(r[0], 6) for r in con.execute("SELECT proba_chain FROM predictions")}
        usados = {round(x, 6) for x, _ in filas}
        assert usados <= chain, "se ajusto sobre la probabilidad servida"


class TestAprendizaje:
    def test_deshace_una_exageracion_conocida(self, con) -> None:
        _sembrar(con, 3000, exagera=3.0)
        assert cc.fit_chain_calibrator(con, min_samples=500) is not None
        cal = cc.load_chain_calibrator(con)

        # La cadena dice 0.6 para vuelos cuya tasa real ronda 0.2.
        assert cal(0.6) < 0.35, f"no corrigio la exageracion: {cal(0.6):.3f}"
        assert cal(0.6) > 0.05

    def test_conserva_el_orden(self, con) -> None:
        """
        Una transformacion monotona no cambia el ranking. Es por esto que
        calibrar no puede subir el AUC, que es el tercer criterio de #7.
        """
        _sembrar(con, 2000)
        cc.fit_chain_calibrator(con, min_samples=500)
        cal = cc.load_chain_calibrator(con)

        entrada = np.linspace(0.0, 1.0, 50)
        salida = cal.transform(entrada)
        assert np.all(np.diff(salida) >= -1e-9), "el calibrador invirtio el orden"

    def test_ignora_las_predicciones_posteriores_al_aterrizaje(self, con) -> None:
        _sembrar(con, 600, fase="POST_LANDING")
        assert cc.fit_chain_calibrator(con, min_samples=100) is None

    def test_ignora_lo_de_fuera_de_la_ventana(self, con) -> None:
        _sembrar(con, 800, hace_dias=90)
        assert cc.fit_chain_calibrator(con, window_days=30, min_samples=100) is None


class TestEstado:
    def test_reporta_edad_y_muestra(self, con) -> None:
        _sembrar(con, 800)
        cc.fit_chain_calibrator(con, min_samples=100)
        estado = cc.calibrator_status(con)
        assert estado["present"] is True
        assert estado["stale"] is False
        assert estado["n_samples"] == 800
        assert estado["age_days"] < 1

    def test_marca_vencido_el_que_lo_esta(self, con) -> None:
        _sembrar(con, 800)
        cc.fit_chain_calibrator(con, min_samples=100)
        viejo = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        con.execute("UPDATE chain_calibrator SET fitted_at_utc = ?", (viejo,))
        con.commit()

        estado = cc.calibrator_status(con)
        assert estado["stale"] is True
        assert "VENCIDO" in estado["detail"]

    def test_guarda_una_sola_fila(self, con) -> None:
        _sembrar(con, 800)
        cc.fit_chain_calibrator(con, min_samples=100)
        cc.fit_chain_calibrator(con, min_samples=100)
        n = con.execute("SELECT COUNT(*) FROM chain_calibrator").fetchone()[0]
        assert n == 1


class TestIndicesDeLaFila:
    """
    `pred_rows` se arma posicionalmente y se inserta con un INSERT explicito.
    Agregar una columna en un lado y no en el otro desplaza todo en silencio: no
    hay error, solo valores en la columna equivocada.

    Paso al sumar `proba_chain`, y este test es la unica forma de que se note.
    """

    def _columnas_del_insert(self) -> list[str]:
        import re
        from pathlib import Path

        fuente = Path(__file__).resolve().parents[1] / "live_pull.py"
        s = fuente.read_text()
        m = re.search(
            r"INSERT OR REPLACE INTO predictions\s*\n\s*\((.*?)\)\s*\n\s*VALUES",
            s, re.S,
        )
        assert m, "no se encontro el INSERT de predictions"
        return [c.strip() for c in m.group(1).replace("\n", " ").split(",") if c.strip()]

    def test_la_cantidad_de_columnas_coincide_con_los_placeholders(self) -> None:
        import re
        from pathlib import Path

        s = (Path(__file__).resolve().parents[1] / "live_pull.py").read_text()
        m = re.search(r"INSERT OR REPLACE INTO predictions.*?VALUES \((.*?)\)", s, re.S)
        placeholders = m.group(1).count("?")
        assert placeholders == len(self._columnas_del_insert()), (
            "el INSERT tiene distinta cantidad de columnas que de placeholders"
        )

    def test_los_indices_nombrados_apuntan_a_la_columna_correcta(self) -> None:
        """Los `I_*` de live_pull tienen que seguir al INSERT."""
        import re
        from pathlib import Path

        s = (Path(__file__).resolve().parents[1] / "live_pull.py").read_text()
        columnas = self._columnas_del_insert()

        esperado = {
            "I_PROBA_FINAL": "proba_delay",
            "I_PROBA_RAW": "proba_raw",
            "I_PROBA_CHAIN": "proba_chain",
            "I_DEP_DELAY": "intermediate_dep_delay_min",
            "I_ADSB_ETA": "adsb_eta_delay_min",
            "I_ADSB_HOLDING": "adsb_holding_min",
        }
        declarados: dict[str, int] = {}
        for linea in re.findall(r"^\s*(I_[A-Z_]+(?:, I_[A-Z_]+)*) = (.+)$", s, re.M):
            nombres = [n.strip() for n in linea[0].split(",")]
            valores = [int(v.strip()) for v in linea[1].split(",")]
            declarados.update(dict(zip(nombres, valores)))

        for nombre, columna in esperado.items():
            assert nombre in declarados, f"falta la constante {nombre}"
            i = declarados[nombre]
            assert columnas[i] == columna, (
                f"{nombre}={i} apunta a `{columnas[i]}`, deberia ser `{columna}`"
            )
