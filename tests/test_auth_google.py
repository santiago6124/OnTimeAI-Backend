"""Google sign-in (issue #8): token exchange, first-login onboarding and profile type."""
from __future__ import annotations

import os
import sqlite3
import uuid

import pytest
from fastapi.testclient import TestClient

import api
from api import app


@pytest.fixture
def client():
    """TestClient with lifespan run, so the users table (and its migration) exists."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def google_email():
    """A throwaway Google identity, removed from users.db afterwards."""
    email = f"test-{uuid.uuid4().hex[:12]}@gmail.com"
    yield email
    con = sqlite3.connect(str(api.USERS_DB_PATH))
    con.execute("DELETE FROM users WHERE username=?", (email,))
    con.commit()
    con.close()


# Cada plataforma usa su propio cliente OAuth y el `aud` del token es el client
# ID de ESA plataforma. Ver Frontend #46.
AUD_WEB = "test-web.apps.googleusercontent.com"
AUD_IOS = "test-ios.apps.googleusercontent.com"
AUD_ANDROID = "test-android.apps.googleusercontent.com"
AUDIENCIAS = (AUD_WEB, AUD_IOS, AUD_ANDROID)


@pytest.fixture
def google_ok(monkeypatch, google_email):
    """Stub del verificador: el token 'valid' resuelve a google_email.

    El falso recibe `(token, request)` y NO una audiencia, porque el codigo ya
    no se la pasa a `verify_oauth2_token`: esa funcion compara `aud` por
    igualdad contra un unico valor y una lista nunca casaria. La comprobacion
    de audiencia la hace `_verify_google_id_token`.
    """
    monkeypatch.setattr(api, "GOOGLE_CLIENT_IDS", AUDIENCIAS)

    def fake_verify(token, request):
        if token != "valid":
            raise ValueError("Token has wrong signature")
        return {"email": google_email, "email_verified": True, "aud": AUD_WEB}

    monkeypatch.setattr("google.oauth2.id_token.verify_oauth2_token", fake_verify)
    return google_email


def _bearer(res):
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


# ── Configuration guard ────────────────────────────────────────────────────

def test_google_login_disabled_without_client_id(client, monkeypatch):
    monkeypatch.setattr(api, "GOOGLE_CLIENT_IDS", ())
    r = client.post("/auth/google", json={"id_token": "whatever"})
    assert r.status_code == 503


# ── Token validation ───────────────────────────────────────────────────────

def test_invalid_id_token_is_rejected(client, google_ok):
    r = client.post("/auth/google", json={"id_token": "forged"})
    assert r.status_code == 401


def test_unverified_email_is_rejected(client, monkeypatch):
    monkeypatch.setattr(api, "GOOGLE_CLIENT_IDS", AUDIENCIAS)
    monkeypatch.setattr(
        "google.oauth2.id_token.verify_oauth2_token",
        lambda token, request: {
            "email": "x@gmail.com", "email_verified": False, "aud": AUD_WEB},
    )
    r = client.post("/auth/google", json={"id_token": "valid"})
    assert r.status_code == 401


# ── First sign-in and onboarding ───────────────────────────────────────────

def test_first_sign_in_creates_user_pending_onboarding(client, google_ok):
    r = client.post("/auth/google", json={"id_token": "valid"})
    assert r.status_code == 200
    body = r.json()
    assert body["is_new_user"] is True
    assert body["user_type"] is None
    assert body["token_type"] == "bearer"

    me = client.get("/auth/me", headers=_bearer(r)).json()
    assert me["username"] == google_ok
    assert me["provider"] == "google"
    assert me["role"] == "user"          # never elevated via this route
    assert me["user_type"] is None


def test_second_sign_in_skips_onboarding(client, google_ok):
    first = client.post("/auth/google", json={"id_token": "valid"})
    client.patch("/users/me", json={"user_type": "b2b"}, headers=_bearer(first))

    second = client.post("/auth/google", json={"id_token": "valid"})
    assert second.json()["is_new_user"] is False
    assert second.json()["user_type"] == "b2b"


def test_google_user_cannot_log_in_with_password(client, google_ok):
    client.post("/auth/google", json={"id_token": "valid"})
    r = client.post("/auth/login", json={"username": google_ok, "password": "anything"})
    assert r.status_code == 401


# ── Profile type (B2B / B2C) ───────────────────────────────────────────────

@pytest.mark.parametrize("user_type", ["b2b", "b2c"])
def test_set_user_type(client, google_ok, user_type):
    login = client.post("/auth/google", json={"id_token": "valid"})
    headers = _bearer(login)

    r = client.patch("/users/me", json={"user_type": user_type}, headers=headers)
    assert r.status_code == 200
    assert r.json()["user_type"] == user_type
    assert client.get("/auth/me", headers=headers).json()["user_type"] == user_type


def test_invalid_user_type_is_rejected(client, google_ok):
    login = client.post("/auth/google", json={"id_token": "valid"})
    r = client.patch("/users/me", json={"user_type": "enterprise"}, headers=_bearer(login))
    assert r.status_code == 400


def test_user_type_requires_authentication(client):
    assert client.patch("/users/me", json={"user_type": "b2b"}).status_code == 401


# ── Existing password accounts keep working ────────────────────────────────

def test_local_login_still_works(client):
    # Las credenciales sembradas salen del entorno (conftest las fija en CI),
    # así que no se pueden hardcodear: en local vienen de .env.local.
    r = client.post(
        "/auth/login",
        json={
            "username": os.environ["API_USERNAME"],
            "password": os.environ["API_PASSWORD"],
        },
    )
    assert r.status_code == 200
    assert client.get("/auth/me", headers=_bearer(r)).json()["provider"] == "local"


# ── Varias audiencias (Frontend #46) ───────────────────────────────────────
#
# Cada plataforma usa su propio cliente OAuth, asi que el `aud` del token es el
# client ID de esa plataforma. Con una sola audiencia permitida, el login
# nativo falla con "ID token de Google invalido" aunque el token sea legitimo.
#
# El cambio mueve la comprobacion de audiencia de `verify_oauth2_token` a
# nuestro codigo, asi que lo que hay que proteger es que siga rechazando: una
# comprobacion que se afloja de mas deja entrar con un token emitido para
# cualquier otra aplicacion de Google.


def _verificador(monkeypatch, aud, email="alguien@gmail.com"):
    monkeypatch.setattr(
        "google.oauth2.id_token.verify_oauth2_token",
        lambda token, request: {"email": email, "email_verified": True, "aud": aud},
    )


@pytest.mark.parametrize("aud", AUDIENCIAS)
def test_acepta_el_token_de_cada_plataforma(client, monkeypatch, aud) -> None:
    monkeypatch.setattr(api, "GOOGLE_CLIENT_IDS", AUDIENCIAS)
    _verificador(monkeypatch, aud, email=f"user-{aud[5:8]}@gmail.com")
    r = client.post("/auth/google", json={"id_token": "valid"})
    assert r.status_code == 200, r.text


def test_rechaza_una_audiencia_que_no_es_nuestra(client, monkeypatch) -> None:
    """Un token legitimo de Google emitido para OTRA aplicacion. Sin esta
    comprobacion serviria para entrar, porque ya no la hace la libreria."""
    monkeypatch.setattr(api, "GOOGLE_CLIENT_IDS", AUDIENCIAS)
    _verificador(monkeypatch, "app-de-otro.apps.googleusercontent.com")
    r = client.post("/auth/google", json={"id_token": "valid"})
    assert r.status_code == 401
    assert "no fue emitido" in r.json()["detail"]


def test_un_token_sin_audiencia_se_rechaza(client, monkeypatch) -> None:
    monkeypatch.setattr(api, "GOOGLE_CLIENT_IDS", AUDIENCIAS)
    monkeypatch.setattr(
        "google.oauth2.id_token.verify_oauth2_token",
        lambda token, request: {"email": "x@gmail.com", "email_verified": True},
    )
    r = client.post("/auth/google", json={"id_token": "valid"})
    assert r.status_code == 401


def test_la_lista_vacia_rechaza_todo_y_no_acepta_cualquier_cosa(client, monkeypatch) -> None:
    """Sin audiencias configuradas el endpoint se deshabilita (503). El riesgo
    al mover la comprobacion es que una lista vacia pase a significar
    "cualquier audiencia sirve"."""
    monkeypatch.setattr(api, "GOOGLE_CLIENT_IDS", ())
    _verificador(monkeypatch, "cualquiera.apps.googleusercontent.com")
    r = client.post("/auth/google", json={"id_token": "valid"})
    assert r.status_code == 503


def test_la_variable_en_singular_sigue_funcionando(monkeypatch) -> None:
    """El entorno desplegado define `GOOGLE_CLIENT_ID`. Si el arranque solo
    leyera `GOOGLE_CLIENT_IDS`, el login con Google se caeria al desplegar."""
    import importlib

    monkeypatch.delenv("GOOGLE_CLIENT_IDS", raising=False)
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "solo-uno.apps.googleusercontent.com")
    recargado = importlib.reload(api)
    try:
        assert recargado.GOOGLE_CLIENT_IDS == ("solo-uno.apps.googleusercontent.com",)
    finally:
        monkeypatch.undo()
        importlib.reload(api)


def test_la_lista_se_separa_por_comas_y_tolera_espacios(monkeypatch) -> None:
    import importlib

    monkeypatch.setenv("GOOGLE_CLIENT_IDS", " a.apps.googleusercontent.com , b.apps.googleusercontent.com ,, ")
    recargado = importlib.reload(api)
    try:
        assert recargado.GOOGLE_CLIENT_IDS == (
            "a.apps.googleusercontent.com", "b.apps.googleusercontent.com")
    finally:
        monkeypatch.undo()
        importlib.reload(api)
