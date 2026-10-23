"""Caracterización de `predict.py`: el frame de inferencia y el CLI de scoring.

CP-COB-02 del trabajo de Calidad (OnTimeAI-Calidad). `prepare_inference_frame`
es la función que arma las features para el modelo tanto en el CLI como en el
pipeline live, con complejidad ciclomática 11 y 42 % de cobertura: sus ramas
—con y sin historial del avión, con y sin lookup de fallback, las features v8
(decaimiento del linaje, PageRank) y v9 (familia de aeronave), y la tolerancia
a fallos de esos módulos opcionales— no estaban caracterizadas. Tampoco `main`,
el CLI que lee un CSV o Parquet, aplica artefacto y calibrador y escribe las
predicciones.

Los frames son sintéticos: el mini-rotación de `tests/test_lineage.py` para
el linaje, y el master sintético de `tests/conftest.py` para entrenar un
artefacto chico con el que ejercitar el CLI de punta a punta.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import predict
from ontimeai.config import TrainConfig
from ontimeai.lineage_fallback import build_lookups, save_lookups
from ontimeai.model import predict_proba, save_artifact
from ontimeai.pipeline import run_training
from predict import prepare_inference_frame
from tests.conftest import make_synthetic_master

RAIZ = Path(__file__).resolve().parent.parent
PAGERANK = json.loads((RAIZ / "artifacts" / "airport_pagerank.json").read_text())


def _rotacion_con_historial() -> pd.DataFrame:
    """Cinco vuelos de dos aviones con horarios reales, como en test_lineage.

    La fila 1 es el segundo vuelo de N1: su anterior aterrizó a las 08:10 y
    este sale a las 10:00, así que el linaje es observable (demora 10, escala
    120 min). La fila 0 es el primer vuelo del día de N1 y no tiene anterior.
    """
    return pd.DataFrame({
        "TAIL_NUM": ["N1", "N1", "N1", "N2", "N1"],
        "OP_CARRIER": ["DL", "DL", "DL", "WN", "DL"],
        "ORIGIN": ["ATL", "MCO", "ATL", "ATL", "LGA"],
        "DEST": ["MCO", "ATL", "LGA", "ORD", "ATL"],
        "FL_DATE": pd.to_datetime(["2024-05-01"] * 4 + ["2024-05-02"]),
        "DAY_OF_WEEK": [3, 3, 3, 3, 4],
        "MONTH": [5, 5, 5, 5, 5],
        "CRS_DEP_MIN": [360, 600, 900, 480, 420],
        "CRS_DEP_TIME": [600, 1000, 1500, 800, 700],
        "EVENT_ORIGIN_UTC": pd.to_datetime([
            "2024-05-01 06:00", "2024-05-01 10:00", "2024-05-01 15:00",
            "2024-05-01 08:00", "2024-05-02 07:00",
        ]),
        "EVENT_DEST_UTC": pd.to_datetime([
            "2024-05-01 08:00", "2024-05-01 12:00", "2024-05-01 17:00",
            "2024-05-01 10:00", "2024-05-02 09:00",
        ]),
        "ARR_DELAY": [10.0, -5.0, 30.0, 45.0, 0.0],
    })


# ── prepare_inference_frame ────────────────────────────────────────────────


class TestFrameDeInferencia:
    def test_con_historial_calcula_el_linaje_del_avion(self) -> None:
        # Arrange: el frame trae ARR_DELAY y los eventos UTC, o sea historial.
        df = _rotacion_con_historial()
        columnas = ["prev_arr_delay_tail", "prev_turnaround_tail_min", "tail_flights_today_prior"]
        # Act
        X = prepare_inference_frame(df, columnas, {})
        # Assert: el segundo vuelo de N1 ve la demora y la escala del anterior.
        assert list(X.columns) == columnas
        assert X.loc[1, "prev_arr_delay_tail"] == 10.0
        assert X.loc[1, "prev_turnaround_tail_min"] == 120.0
        assert X.loc[1, "tail_flights_today_prior"] == 1
        assert pd.isna(X.loc[0, "prev_arr_delay_tail"])

    def test_el_objetivo_nunca_llega_al_modelo_aunque_se_lo_pida(self) -> None:
        # Arrange: alguien mete ARR_DELAY en las feature_cols.
        df = _rotacion_con_historial()
        # Act
        X = prepare_inference_frame(df, ["ARR_DELAY", "prev_arr_delay_tail"], {})
        # Assert: la columna existe pero vacía; el valor real se descartó.
        assert X["ARR_DELAY"].isna().all()
        assert X.loc[1, "prev_arr_delay_tail"] == 10.0

    def test_sin_historial_el_linaje_queda_vacio(self) -> None:
        # Un CSV de scoring real no trae ARR_DELAY ni los eventos UTC.
        df = _rotacion_con_historial().drop(columns=["ARR_DELAY", "EVENT_ORIGIN_UTC", "EVENT_DEST_UTC"])
        X = prepare_inference_frame(df, ["prev_arr_delay_tail", "CRS_DEP_MIN"], {})
        assert X["prev_arr_delay_tail"].isna().all()
        assert X["CRS_DEP_MIN"].tolist() == [360, 600, 900, 480, 420]

    def test_el_fallback_rellena_el_linaje_que_falta_sin_pisar_el_que_hay(self, tiny_master) -> None:
        # Arrange: lookups fríos del master sintético.
        lookups = build_lookups(tiny_master)
        df = _rotacion_con_historial()
        # Act
        sin = prepare_inference_frame(df, ["prev_arr_delay_tail"], {})
        con = prepare_inference_frame(df, ["prev_arr_delay_tail"], {}, fallback_lookup=lookups)
        # Assert: la fila 0 (sin anterior) se imputa; la fila 1 conserva su 10.0.
        assert pd.isna(sin.loc[0, "prev_arr_delay_tail"])
        assert not pd.isna(con.loc[0, "prev_arr_delay_tail"])
        assert con.loc[1, "prev_arr_delay_tail"] == 10.0

    def test_el_decaimiento_del_linaje_se_calcula_desde_la_escala(self) -> None:
        # Arrange: v8 pide TAIL_DELAY_DECAY = demora_anterior * exp(-escala/60).
        df = _rotacion_con_historial()
        # Act
        X = prepare_inference_frame(df, ["TAIL_DELAY_DECAY"], {})
        # Assert: fila 1 → 10 * exp(-120/60).
        assert X.loc[1, "TAIL_DELAY_DECAY"] == pytest.approx(10.0 * np.exp(-2.0))
        assert pd.isna(X.loc[0, "TAIL_DELAY_DECAY"])

    def test_el_pagerank_se_une_desde_el_artefacto_del_repo(self) -> None:
        # Arrange: ATL está en el lookup; "XXX" no.
        df = _rotacion_con_historial().drop(columns=["ARR_DELAY"])
        df.loc[0, "DEST"] = "XXX"
        # Act
        X = prepare_inference_frame(df, ["ORIGIN_PAGERANK", "DEST_PAGERANK"], {})
        # Assert
        assert X.loc[0, "ORIGIN_PAGERANK"] == pytest.approx(PAGERANK["ATL"])
        assert X.loc[0, "DEST_PAGERANK"] == 0.0
        assert X.loc[1, "DEST_PAGERANK"] == pytest.approx(PAGERANK["ATL"])

    def test_si_las_features_v8_fallan_el_frame_sale_igual(self, monkeypatch) -> None:
        # Arrange: el módulo v8 explota al calcular el decaimiento.
        import feature_engineering_v7.v8_features as v8

        def _explota(df):
            raise RuntimeError("v8 roto")

        monkeypatch.setattr(v8, "add_tail_delay_decay", _explota)
        df = _rotacion_con_historial()
        # Act
        X = prepare_inference_frame(df, ["TAIL_DELAY_DECAY", "CRS_DEP_MIN"], {})
        # Assert: no propaga; la columna queda vacía y el resto intacto.
        assert X["TAIL_DELAY_DECAY"].isna().all()
        assert X["CRS_DEP_MIN"].tolist() == [360, 600, 900, 480, 420]

    def test_la_familia_de_aeronave_sale_del_lookup_o_de_la_aerolinea(self) -> None:
        # Arrange: un tail conocido del lookup, uno desconocido de Southwest
        # (flota 737) y uno desconocido de Delta (sin regla por aerolínea).
        lookup = json.loads((RAIZ / "artifacts" / "tail_to_aircraft_family.json").read_text())
        conocido = next(t for t, f in lookup.items() if t.startswith("N") and f == "B737_FAMILY")
        df = pd.DataFrame({
            "TAIL_NUM": [conocido, "N-NUNCA-VISTO", "N-TAMPOCO"],
            "OP_CARRIER": ["DL", "WN", "DL"],
            "ORIGIN": ["ATL"] * 3, "DEST": ["MCO"] * 3,
            "FL_DATE": pd.to_datetime(["2024-05-01"] * 3),
            "CRS_DEP_MIN": [600, 610, 620], "DAY_OF_WEEK": [3] * 3, "MONTH": [5] * 3,
        })
        # Act
        X = prepare_inference_frame(df, ["AIRCRAFT_FAMILY"], {})
        # Assert
        assert X["AIRCRAFT_FAMILY"].tolist() == ["B737_FAMILY", "B737_FAMILY", "OTHER"]

    def test_si_la_familia_no_se_puede_calcular_queda_other(self, monkeypatch) -> None:
        import feature_engineering_v7.aircraft_type as at

        def _explota(*args, **kwargs):
            raise FileNotFoundError("sin lookup")

        monkeypatch.setattr(at, "add_aircraft_family", _explota)
        df = _rotacion_con_historial()
        X = prepare_inference_frame(df, ["AIRCRAFT_FAMILY"], {})
        assert X["AIRCRAFT_FAMILY"].tolist() == ["OTHER"] * 5

    def test_una_feature_que_el_frame_no_trae_se_crea_vacia_y_en_orden(self) -> None:
        df = _rotacion_con_historial()
        columnas = ["NO_EXISTE", "CRS_DEP_MIN", "OTRA_QUE_FALTA"]
        X = prepare_inference_frame(df, columnas, {})
        assert list(X.columns) == columnas
        assert X["NO_EXISTE"].isna().all() and X["OTRA_QUE_FALTA"].isna().all()

    def test_el_mapeo_categorico_se_aplica_al_final(self) -> None:
        df = _rotacion_con_historial()
        X = prepare_inference_frame(df, ["OP_CARRIER"], {"OP_CARRIER": ["DL", "AA"]})
        assert str(X["OP_CARRIER"].dtype) == "category"
        assert list(X["OP_CARRIER"].cat.categories) == ["DL", "AA"]
        # WN no está en el mapeo del artefacto: queda NaN, como en entrenamiento.
        assert pd.isna(X.loc[3, "OP_CARRIER"])


# ── main (CLI) ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def artefactos(tmp_path_factory):
    """Un artefacto binario y uno multiclase chicos, entrenados una sola vez."""
    base = tmp_path_factory.mktemp("predict-cli")
    master = make_synthetic_master(n=2000, seed=7)
    binario = run_training(
        TrainConfig(target="binary", num_boost_round=30, early_stopping_rounds=5,
                    artifacts_dir=base / "binario"),
        df_raw=master,
    )
    multiclase = run_training(
        TrainConfig(target="multiclass", num_boost_round=30, early_stopping_rounds=5,
                    artifacts_dir=base / "multiclase"),
        df_raw=master,
    )
    assert binario.artifact_dir is not None and multiclase.artifact_dir is not None
    # El mismo booster binario, guardado sin calibrador: el CLI tiene que
    # servir la probabilidad cruda cuando el artefacto no trae uno.
    sin_calibrador = save_artifact(
        binario.booster, threshold=binario.threshold, feature_cols=binario.feature_cols,
        cat_cols=binario.cat_cols, cat_mapping=binario.cat_mapping, target="binary",
        metadata={}, out_dir=base / "sin-calibrador", calibrator=None,
    )
    return {"base": base, "master": master, "binario": binario.artifact_dir,
            "multiclase": multiclase.artifact_dir, "sin_calibrador": sin_calibrador}


def _csv_de_scoring(master: pd.DataFrame, destino: Path, n: int = 80) -> Path:
    """Un extracto BTS para puntuar, con la forma del master de entrenamiento.

    Trae ARR_DELAY y los eventos UTC: el CLI los usa para calcular el linaje
    y después descarta el objetivo como fuga. Ver
    `test_sin_historial_el_cli_no_puede_puntuar` para lo que pasa sin ellos.
    """
    master.head(n).to_csv(destino, index=False)
    return destino


def _correr(artefacto: Path, entrada: Path, salida: Path, fallback: Path) -> pd.DataFrame:
    codigo = predict.main([
        "--artifact", str(artefacto), "--input", str(entrada),
        "--output", str(salida), "--fallback-lookup", str(fallback),
    ])
    assert codigo == 0
    return pd.read_csv(salida)


class TestCli:
    def test_escribe_una_probabilidad_y_una_etiqueta_por_vuelo(self, artefactos, tmp_path) -> None:
        # Arrange
        entrada = _csv_de_scoring(artefactos["master"], tmp_path / "vuelos.csv")
        meta = joblib.load(artefactos["binario"] / "meta.joblib")
        # Act
        out = _correr(artefactos["binario"], entrada, tmp_path / "pred.csv", tmp_path / "no-existe.joblib")
        # Assert: una fila por vuelo, probabilidad válida y etiqueta coherente
        # con el umbral guardado en el artefacto.
        assert list(out.columns) == ["proba_delay", "predicted_delay"]
        assert len(out) == 80
        assert out["proba_delay"].between(0.0, 1.0).all()
        assert set(out["predicted_delay"].unique()) <= {0, 1}
        esperado = (out["proba_delay"] >= meta["threshold"]).astype(int)
        assert out["predicted_delay"].tolist() == esperado.tolist()

    def test_la_probabilidad_pasa_por_el_calibrador_del_artefacto(self, artefactos, tmp_path) -> None:
        # Arrange: el artefacto binario se entrenó con calibración isotónica.
        meta = joblib.load(artefactos["binario"] / "meta.joblib")
        assert meta["calibrator"] is not None
        entrada = _csv_de_scoring(artefactos["master"], tmp_path / "vuelos.csv")
        # Act
        out = _correr(artefactos["binario"], entrada, tmp_path / "pred.csv", tmp_path / "nope.joblib")
        # Assert: un calibrador isotónico produce pocos valores distintos
        # (escalones); la salida cruda del booster daría 80 valores únicos.
        assert out["proba_delay"].nunique() < 80

    def test_sin_calibrador_la_probabilidad_es_la_cruda_del_booster(self, artefactos, tmp_path) -> None:
        # Arrange: el mismo frame que arma el CLI, puntuado a mano con el booster.
        entrada = _csv_de_scoring(artefactos["master"], tmp_path / "vuelos.csv", n=30)
        from ontimeai.model import load_artifact

        meta = load_artifact(artefactos["sin_calibrador"])
        assert meta["calibrator"] is None
        crudo = pd.read_csv(entrada, parse_dates=["FL_DATE"], low_memory=False)
        X = prepare_inference_frame(crudo, meta["feature_cols"], meta["cat_mapping"])
        esperado = predict_proba(meta["booster"], X)
        # Act
        out = _correr(artefactos["sin_calibrador"], entrada, tmp_path / "pred.csv", tmp_path / "nope.joblib")
        # Assert: sin calibrador, el CLI escribe exactamente lo que dice el booster.
        np.testing.assert_allclose(out["proba_delay"].to_numpy(), esperado, atol=1e-9)

    def test_sin_historial_el_cli_no_puede_puntuar(self, artefactos, tmp_path) -> None:
        # Caracterización de un defecto, no de un deseo: un CSV sin ARR_DELAY
        # ni eventos UTC (el caso de vuelos futuros) hace que las features de
        # linaje se creen con `pd.NA` en dtype object, y LightGBM las rechaza.
        # El pipeline live no pasa por acá porque arma el linaje por su cuenta.
        # Si alguien lo arregla, este test tiene que cambiar a propósito.
        entrada = tmp_path / "futuros.csv"
        artefactos["master"].head(5).drop(
            columns=["ARR_DELAY", "EVENT_ORIGIN_UTC", "EVENT_DEST_UTC"],
        ).to_csv(entrada, index=False)
        with pytest.raises(ValueError, match="dtypes"):
            predict.main([
                "--artifact", str(artefactos["binario"]), "--input", str(entrada),
                "--output", str(tmp_path / "pred.csv"),
                "--fallback-lookup", str(tmp_path / "nope.joblib"),
            ])

    def test_acepta_parquet_y_lee_la_fecha_como_fecha(self, artefactos, tmp_path) -> None:
        # Arrange: el Parquet conserva los tipos (fechas como fechas); el CSV
        # los pierde y el CLI tiene que reconstruirlos al leer.
        entrada = tmp_path / "vuelos.parquet"
        artefactos["master"].head(50).to_parquet(entrada, index=False)
        # Act
        out = _correr(artefactos["binario"], entrada, tmp_path / "pred.csv", tmp_path / "nope.joblib")
        # Assert
        assert len(out) == 50
        assert out["proba_delay"].between(0.0, 1.0).all()

    def test_multiclase_escribe_una_columna_por_clase(self, artefactos, tmp_path) -> None:
        # Arrange
        entrada = _csv_de_scoring(artefactos["master"], tmp_path / "vuelos.csv", n=40)
        # Act
        out = _correr(artefactos["multiclase"], entrada, tmp_path / "pred.csv", tmp_path / "nope.joblib")
        # Assert: cuatro clases, filas que suman 1 y la etiqueta es el argmax.
        probas = [f"proba_class_{i}" for i in range(4)]
        assert list(out.columns) == probas + ["predicted_class"]
        assert len(out) == 40
        np.testing.assert_allclose(out[probas].sum(axis=1), 1.0, atol=1e-6)
        assert out["predicted_class"].tolist() == out[probas].to_numpy().argmax(axis=1).tolist()

    def test_carga_el_fallback_cuando_el_archivo_existe(self, artefactos, tmp_path, capsys) -> None:
        # Arrange
        fallback = save_lookups(build_lookups(artefactos["master"]), tmp_path / "fallback.joblib")
        entrada = _csv_de_scoring(artefactos["master"], tmp_path / "vuelos.csv", n=20)
        # Act
        out = _correr(artefactos["binario"], entrada, tmp_path / "pred.csv", fallback)
        # Assert
        assert len(out) == 20
        assert "Loaded cold-deck fallback" in capsys.readouterr().out

    def test_sin_archivo_de_fallback_sigue_en_silencio(self, artefactos, tmp_path, capsys) -> None:
        entrada = _csv_de_scoring(artefactos["master"], tmp_path / "vuelos.csv", n=20)
        _correr(artefactos["binario"], entrada, tmp_path / "pred.csv", tmp_path / "no-existe.joblib")
        salida = capsys.readouterr().out
        assert "Loaded cold-deck fallback" not in salida
        assert "Wrote 20 predictions" in salida
