"""
La guarda de /auth/google contra cuentas locales que reclaman un correo.

El alta propia ya no existe —Firebase se ocupa de las credenciales por correo y
verifica que sean de quien las presenta, ver tests/test_auth_firebase.py— asi
que hoy solo un administrador puede crear una cuenta local cuyo usuario sea un
correo.

La guarda se queda igual: mientras esa forma de cuenta sea posible, entregarsela
a quien llegue con un token de Google seria darle acceso a una cuenta cuya
contrasena puso otro.
"""
from __future__ import annotations

import shutil

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import api

    tmp = tmp_path_factory.mktemp("guarda")
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


class TestGoogle:
    def test_una_cuenta_de_google_vuelve_a_entrar(self, client, monkeypatch) -> None:
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

    def test_una_cuenta_administrativa_se_vincula(self, client, monkeypatch) -> None:
        """
        `admin` y `viewer` no se parecen a un correo, asi que la guarda no las
        alcanza y siguen pudiendo vincularse con Google.
        """
        import api

        monkeypatch.setattr(
            api, "_verify_google_id_token",
            lambda _t: {"email": "otro@gmail.com", "email_verified": True},
        )
        assert client.post("/auth/google", json={"id_token": "t"}).status_code == 200

    def test_no_entrega_una_cuenta_local_que_reclama_ese_correo(
        self, client, monkeypatch
    ) -> None:
        """
        Hoy solo un administrador puede crear esa forma de cuenta, pero
        mientras sea posible la guarda tiene que estar: su contrasena la puso
        alguien que no es el duenio del correo.
        """
        import api

        correo = "disputada@gmail.com"
        con = api._get_users_con()
        con.execute(
            "INSERT INTO users (username, password_hash, role, provider) "
            "VALUES (?,?,'user','local')",
            (correo, api._hash_password("la-que-puso-otro")),
        )
        con.commit()
        con.close()

        monkeypatch.setattr(
            api, "_verify_google_id_token",
            lambda _t: {"email": correo, "email_verified": True},
        )
        r = client.post("/auth/google", json={"id_token": "t"})
        assert r.status_code == 409
        assert "contraseña" in r.json()["detail"]
