"""Google sign-in (issue #8): token exchange, first-login onboarding and profile type."""
from __future__ import annotations

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


@pytest.fixture
def google_ok(monkeypatch, google_email):
    """Stub Google's verifier so a token literally named 'valid' resolves to google_email."""
    monkeypatch.setattr(api, "GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")

    def fake_verify(token, request, audience):
        if token != "valid":
            raise ValueError("Token has wrong audience")
        return {"email": google_email, "email_verified": True, "aud": audience}

    monkeypatch.setattr("google.oauth2.id_token.verify_oauth2_token", fake_verify)
    return google_email


def _bearer(res):
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


# ── Configuration guard ────────────────────────────────────────────────────

def test_google_login_disabled_without_client_id(client, monkeypatch):
    monkeypatch.setattr(api, "GOOGLE_CLIENT_ID", "")
    r = client.post("/auth/google", json={"id_token": "whatever"})
    assert r.status_code == 503


# ── Token validation ───────────────────────────────────────────────────────

def test_invalid_id_token_is_rejected(client, google_ok):
    r = client.post("/auth/google", json={"id_token": "forged"})
    assert r.status_code == 401


def test_unverified_email_is_rejected(client, monkeypatch):
    monkeypatch.setattr(api, "GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    monkeypatch.setattr(
        "google.oauth2.id_token.verify_oauth2_token",
        lambda token, request, audience: {"email": "x@gmail.com", "email_verified": False},
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
    r = client.post("/auth/login", json={"username": "admin", "password": "ontimeai2026"})
    assert r.status_code == 200
    assert client.get("/auth/me", headers=_bearer(r)).json()["provider"] == "local"
