"""
Alta propia con correo y contrasena.

El sistema ya era abierto —cualquiera con cuenta de Google se registraba solo—
asi que abrir el otro metodo no cambia la superficie: quita la asimetria de que
uno dependiera de un superadmin y el otro no.

Lo que si introduce es un riesgo, y la mitad de estos tests son sobre eso.
`/auth/google` busca la cuenta con `WHERE username=? OR email=?`, y el alta
propia guarda el correo como nombre de usuario sin poder verificarlo. Sin
guarda, alguien se registra con el correo ajeno y espera: cuando el duenio
entra con Google, cae en la cuenta del otro, cuya contrasena el otro conoce.
"""
from __future__ import annotations

import os
import shutil

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import api

    tmp = tmp_path_factory.mktemp("registro")
    originales = (api.DB_PATH, api.USERS_DB_PATH)
    db = tmp / "live_data.db"
    shutil.copy(api.DB_PATH, db)
    api.DB_PATH = db
    api.USERS_DB_PATH = tmp / "users.db"
    try:
        with TestClient(api.app) as c:
            yield c
    finally:
        api.DB_PATH, api.USERS_DB_PATH = originales


def _registrar(client, email, password="contrasena-larga"):
    return client.post("/auth/register", json={"email": email, "password": password})


class TestAlta:
    def test_crea_la_cuenta_y_devuelve_token(self, client) -> None:
        r = _registrar(client, "alguien@ejemplo.com")
        assert r.status_code == 201
        cuerpo = r.json()
        assert cuerpo["access_token"]
        assert cuerpo["is_new_user"] is True
        assert cuerpo["user_type"] is None, "el tipo se elige en el onboarding"

    def test_la_cuenta_nueva_puede_entrar_con_su_contrasena(self, client) -> None:
        _registrar(client, "entra@ejemplo.com", "una-contrasena-larga")
        r = client.post(
            "/auth/login",
            json={"username": "entra@ejemplo.com", "password": "una-contrasena-larga"},
        )
        assert r.status_code == 200

    def test_normaliza_el_correo(self, client) -> None:
        assert _registrar(client, "  MAYUS@Ejemplo.COM  ").status_code == 201
        # El mismo correo en otra caja ya no entra.
        assert _registrar(client, "mayus@ejemplo.com").status_code == 409

    def test_un_correo_repetido_da_409(self, client) -> None:
        _registrar(client, "repetido@ejemplo.com")
        r = _registrar(client, "repetido@ejemplo.com")
        assert r.status_code == 409
        assert "ya existe" in r.json()["detail"].lower()


class TestValidacion:
    @pytest.mark.parametrize(
        "correo",
        ["sinarroba.com", "@sinlocal.com", "sin@dominio", "dos@@arrobas.com",
         "con espacio@ejemplo.com", ""],
    )
    def test_rechaza_correos_invalidos(self, client, correo) -> None:
        assert _registrar(client, correo).status_code == 400

    def test_rechaza_contrasenas_cortas(self, client) -> None:
        r = _registrar(client, "corta@ejemplo.com", "123456789")
        assert r.status_code == 400
        assert "caracteres" in r.json()["detail"]

    def test_acepta_el_largo_minimo_exacto(self, client) -> None:
        assert _registrar(client, "justa@ejemplo.com", "1234567890").status_code == 201


class TestNoSeElevaElRol:
    def test_el_alta_propia_siempre_es_user(self, client) -> None:
        """
        El rol no lo elige quien se registra. Si `role` fuera parte del cuerpo,
        cualquiera se daria de alta como superadmin.
        """
        r = client.post(
            "/auth/register",
            json={
                "email": "aspirante@ejemplo.com",
                "password": "contrasena-larga",
                "role": "superadmin",
            },
        )
        assert r.status_code == 201

        # El token no sirve para un endpoint de superadmin.
        token = r.json()["access_token"]
        assert client.get(
            "/admin/users", headers={"Authorization": f"Bearer {token}"}
        ).status_code == 403


class TestTomaDeCuenta:
    """
    El riesgo que abre el registro propio, y su guarda.

    Sin verificacion de correo, el alta propia no prueba que el correo sea de
    quien se registra. La guarda vive en /auth/google.
    """

    def test_google_no_entra_a_una_cuenta_local_con_ese_correo(
        self, client, monkeypatch
    ) -> None:
        import api

        victima = "victima@gmail.com"
        _registrar(client, victima, "la-que-sabe-el-atacante")

        # El duenio real llega con un token de Google verificado por Google.
        monkeypatch.setattr(
            api, "_verify_google_id_token",
            lambda _t: {"email": victima, "email_verified": True},
        )
        r = client.post("/auth/google", json={"id_token": "token-valido"})

        assert r.status_code == 409, (
            "Google entrego una cuenta local que reclamaba ese correo sin probarlo"
        )
        assert "contraseña" in r.json()["detail"]

    def test_una_cuenta_de_google_si_vuelve_a_entrar(self, client, monkeypatch) -> None:
        import api

        correo = "propia@gmail.com"
        monkeypatch.setattr(
            api, "_verify_google_id_token",
            lambda _t: {"email": correo, "email_verified": True},
        )
        primera = client.post("/auth/google", json={"id_token": "t"})
        assert primera.status_code == 200
        assert primera.json()["is_new_user"] is True

        segunda = client.post("/auth/google", json={"id_token": "t"})
        assert segunda.status_code == 200
        assert segunda.json()["is_new_user"] is False

    def test_una_cuenta_administrativa_no_se_confunde(self, client, monkeypatch) -> None:
        """
        La guarda compara el nombre de usuario con el correo. Las cuentas
        creadas por un administrador —`admin`, `viewer`— no se parecen a un
        correo, asi que siguen pudiendo vincularse con Google como antes.
        """
        import api

        monkeypatch.setattr(
            api, "_verify_google_id_token",
            lambda _t: {"email": "otro@gmail.com", "email_verified": True},
        )
        r = client.post("/auth/google", json={"id_token": "t"})
        assert r.status_code == 200
