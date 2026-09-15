"""
Acceso via Firebase Authentication.

Firebase se ocupa de las credenciales —alta, verificacion del correo,
recuperacion de contrasena— y esta tabla sigue siendo duenia del rol y del tipo
de cuenta. Es el mismo reparto que con /auth/google, con una diferencia que lo
cambia todo: aca el correo llega **probado**.

Eso permite resolver algo que la guarda de /auth/google solo podia bloquear. El
alta propia no puede verificar el correo, asi que cualquiera pudo registrar uno
ajeno y conocer su contrasena. Cuando el duenio real aparece con el correo
probado, la cuenta pasa a ser suya y la contrasena del otro deja de servir.
"""
from __future__ import annotations

import shutil

import pytest
from fastapi.testclient import TestClient

PROYECTO = "ontimeai-prod"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import api

    tmp = tmp_path_factory.mktemp("firebase")
    originales = (api.DB_PATH, api.USERS_DB_PATH, api.FIREBASE_PROJECT_ID)
    db = tmp / "live_data.db"
    shutil.copy(api.DB_PATH, db)
    api.DB_PATH = db
    api.USERS_DB_PATH = tmp / "users.db"
    api.FIREBASE_PROJECT_ID = PROYECTO
    try:
        with TestClient(api.app) as c:
            yield c
    finally:
        api.DB_PATH, api.USERS_DB_PATH, api.FIREBASE_PROJECT_ID = originales


def _con_token(monkeypatch, email):
    """Sustituye la validacion criptografica por los claims que interesan.

    Se reemplaza la funcion entera y no solo la llamada a google-auth porque
    estos tests miran que hace el endpoint con un correo ya probado. La logica
    de validacion en si tiene sus propios tests mas abajo.
    """
    import api

    monkeypatch.setattr(
        api,
        "_verify_firebase_id_token",
        lambda _t: {"email": email, "email_verified": True},
    )


def _entrar(client):
    return client.post("/auth/firebase", json={"id_token": "token"})


def _cuenta_local(correo, clave):
    """Crea una cuenta local cuyo usuario ES un correo.

    Se inserta directo porque el alta propia ya no existe: Firebase se ocupa de
    las credenciales por correo. Hoy solo un administrador puede dejar una
    cuenta con esta forma, pero mientras sea posible el endpoint tiene que
    resolverla bien.
    """
    import api

    con = api._get_users_con()
    con.execute(
        "INSERT INTO users (username, password_hash, role, provider) "
        "VALUES (?,?,'user','local')",
        (correo, api._hash_password(clave)),
    )
    con.commit()
    con.close()


class TestValidacionDelToken:
    """
    La logica de `_verify_firebase_id_token`, con la verificacion criptografica
    de google-auth sustituida. Lo que se prueba es que hacemos con los claims.
    """

    def _con_claims(self, monkeypatch, claims):
        import api
        from google.oauth2 import id_token as google_id_token

        monkeypatch.setattr(api, "FIREBASE_PROJECT_ID", PROYECTO)
        monkeypatch.setattr(
            google_id_token, "verify_firebase_token", lambda *a, **k: claims
        )

    def test_rechaza_un_correo_sin_verificar(self, monkeypatch) -> None:
        """
        Es el punto de todo el endpoint. Sin esta exigencia no aporta nada
        sobre el alta propia, que tampoco puede verificar el correo.
        """
        import api

        self._con_claims(monkeypatch, {"email": "a@b.com", "email_verified": False})
        with pytest.raises(api.HTTPException) as exc:
            api._verify_firebase_id_token("token")
        assert exc.value.status_code == 403
        assert "verificar" in exc.value.detail.lower()

    def test_rechaza_un_token_sin_correo(self, monkeypatch) -> None:
        import api

        self._con_claims(monkeypatch, {"email_verified": True})
        with pytest.raises(api.HTTPException) as exc:
            api._verify_firebase_id_token("token")
        assert exc.value.status_code == 401

    def test_acepta_un_correo_verificado(self, monkeypatch) -> None:
        import api

        self._con_claims(monkeypatch, {"email": "a@b.com", "email_verified": True})
        assert api._verify_firebase_id_token("token")["email"] == "a@b.com"

    def test_un_token_invalido_da_401(self, monkeypatch) -> None:
        import api
        from google.oauth2 import id_token as google_id_token

        monkeypatch.setattr(api, "FIREBASE_PROJECT_ID", PROYECTO)

        def _falla(*a, **k):
            raise ValueError("Token expired")

        monkeypatch.setattr(google_id_token, "verify_firebase_token", _falla)
        with pytest.raises(api.HTTPException) as exc:
            api._verify_firebase_id_token("token")
        assert exc.value.status_code == 401


class TestCorreoVerificado:
    def test_crea_la_cuenta_la_primera_vez(self, client, monkeypatch) -> None:
        _con_token(monkeypatch, "nueva@ejemplo.com")
        r = _entrar(client)
        assert r.status_code == 200
        assert r.json()["is_new_user"] is True

    def test_la_segunda_vez_no_es_nueva(self, client, monkeypatch) -> None:
        _con_token(monkeypatch, "repite@ejemplo.com")
        assert _entrar(client).json()["is_new_user"] is True
        segunda = _entrar(client).json()
        assert segunda["is_new_user"] is False


class TestTransferenciaDeCuenta:
    """
    Lo que Firebase permite y la guarda de /auth/google solo podia bloquear.
    """

    def test_el_duenio_verificado_se_queda_con_la_cuenta(
        self, client, monkeypatch
    ) -> None:
        correo = "disputada@ejemplo.com"
        # Alguien la ocupo antes con una contrasena que conoce.
        _cuenta_local(correo, "la-del-ocupante")

        _con_token(monkeypatch, correo)
        r = _entrar(client)
        assert r.status_code == 200, "el duenio verificado no pudo entrar"

    def test_la_contrasena_del_ocupante_deja_de_servir(
        self, client, monkeypatch
    ) -> None:
        correo = "invalidada@ejemplo.com"
        clave = "la-del-ocupante"
        _cuenta_local(correo, clave)
        # Antes de la transferencia, esa contrasena entra.
        assert client.post(
            "/auth/login", json={"username": correo, "password": clave}
        ).status_code == 200

        _con_token(monkeypatch, correo)
        _entrar(client)

        assert client.post(
            "/auth/login", json={"username": correo, "password": clave}
        ).status_code == 401, (
            "el ocupante sigue entrando a una cuenta que ya no es suya"
        )

    def test_conserva_el_rol_y_el_tipo_de_la_cuenta(
        self, client, monkeypatch
    ) -> None:
        """La transferencia cambia quien entra, no que permisos tiene."""
        correo = "conserva@ejemplo.com"
        _cuenta_local(correo, "contrasena-larga")

        _con_token(monkeypatch, correo)
        cuerpo = _entrar(client).json()
        token = cuerpo["access_token"]
        # Sigue siendo `user`: no gana permisos por entrar con Firebase.
        assert client.get(
            "/admin/users", headers={"Authorization": f"Bearer {token}"}
        ).status_code == 403


class TestCuentasAdministrativas:
    def test_no_le_invalida_la_contrasena_a_una_cuenta_de_admin(
        self, client, monkeypatch
    ) -> None:
        """
        `admin` y `viewer` las creo alguien de confianza y su contrasena no se
        toca: la transferencia solo aplica cuando el nombre de usuario ES el
        correo, que es la forma que deja el alta propia.
        """
        import os

        usuario = os.environ["API_USERNAME"]
        clave = os.environ["API_PASSWORD"]

        _con_token(monkeypatch, "cualquier@ejemplo.com")
        _entrar(client)

        assert client.post(
            "/auth/login", json={"username": usuario, "password": clave}
        ).status_code == 200


class TestSinConfigurar:
    def test_sin_proyecto_el_endpoint_avisa(self, client, monkeypatch) -> None:
        import api

        monkeypatch.setattr(api, "FIREBASE_PROJECT_ID", "")
        r = _entrar(client)
        assert r.status_code == 503
        assert "Firebase" in r.json()["detail"]
