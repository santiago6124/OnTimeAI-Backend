"""Tabla de decisión de la autenticación y la autorización (CP-AUTH-01..10).

Técnica de caja negra: las condiciones son "el usuario existe", "está activo"
y "la contraseña es correcta" para el login; y "hay token", "el token es
válido", "no venció" y "el rol es superadmin" para las rutas de administración.
Cada fila de la tabla es un test; el resultado esperado quedó escrito antes de
ejecutar (ver OnTimeAI-Calidad/docs/CASOS_DE_PRUEBA.md).

| CP | existe | activo | contraseña | esperado |
|---|---|---|---|---|
| AUTH-01 | sí | sí | correcta | 200 con token y rol |
| AUTH-02 | no | —  | —        | 401 |
| AUTH-03 | sí | no | correcta | 401 |
| AUTH-04 | sí | sí | incorrecta | 401 |

| CP | token | válido | vigente | superadmin | esperado en /admin/users |
|---|---|---|---|---|---|
| AUTH-05 | no | — | — | — | 401 |
| AUTH-06 | sí | no (firma ajena) | — | — | 401 |
| AUTH-07 | sí | sí | no (vencido) | sí | 401 |
| AUTH-08 | sí | sí | sí | no (viewer) | 403 |
| AUTH-09 | sí | sí | sí | sí | 200 |
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from jose import jwt


@pytest.fixture(scope="module")
def entorno(tmp_path_factory):
    """TestClient sin autenticar, con bases temporales (mismo patrón que test_api.py)."""
    import api

    tmp = tmp_path_factory.mktemp("auth")
    originales = (api.DB_PATH, api.USERS_DB_PATH)
    db = tmp / "live_data.db"
    shutil.copy(api.DB_PATH, db)
    api.DB_PATH = db
    api.USERS_DB_PATH = tmp / "users.db"
    try:
        with TestClient(api.app) as c:
            yield c, api
    finally:
        api.DB_PATH, api.USERS_DB_PATH = originales


def _login(c, usuario, clave):
    return c.post("/auth/login", json={"username": usuario, "password": clave})


def _token_admin(c):
    r = _login(c, os.environ["API_USERNAME"], os.environ["API_PASSWORD"])
    assert r.status_code == 200
    return r.json()["access_token"]


# ── Login ───────────────────────────────────────────────────────────────────

def test_auth_01_usuario_activo_con_clave_correcta_recibe_token(entorno):
    c, _ = entorno
    r = _login(c, os.environ["API_USERNAME_VIEWER"], os.environ["API_PASSWORD_VIEWER"])
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["token_type"] == "bearer" and cuerpo["access_token"]
    assert jwt.get_unverified_claims(cuerpo["access_token"])["role"] == "user"


def test_auth_02_usuario_inexistente_es_401(entorno):
    c, _ = entorno
    r = _login(c, "no-existe", "da-igual")
    assert r.status_code == 401


def test_auth_03_usuario_desactivado_es_401_aunque_la_clave_sea_correcta(entorno):
    c, _ = entorno
    admin = {"Authorization": f"Bearer {_token_admin(c)}"}
    # Arrange: un usuario nuevo, que entra, y después se desactiva
    assert c.post("/admin/users", json={"username": "cp-auth-03", "password": "clave-03"}, headers=admin).status_code == 201
    assert _login(c, "cp-auth-03", "clave-03").status_code == 200
    assert c.patch("/admin/users/cp-auth-03", json={"active": False}, headers=admin).status_code == 200
    # Act / Assert
    assert _login(c, "cp-auth-03", "clave-03").status_code == 401


def test_auth_04_clave_incorrecta_es_401(entorno):
    c, _ = entorno
    r = _login(c, os.environ["API_USERNAME_VIEWER"], "otra-clave")
    assert r.status_code == 401


# ── Autorización en rutas de administración ─────────────────────────────────

def test_auth_05_sin_token_es_401(entorno):
    c, _ = entorno
    assert c.get("/admin/users").status_code == 401


def test_auth_06_token_con_firma_ajena_es_401(entorno):
    c, api = entorno
    ajeno = jwt.encode({"sub": "x", "role": "superadmin",
                        "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
                       "otro-secreto", algorithm=api.JWT_ALGORITHM)
    assert c.get("/admin/users", headers={"Authorization": f"Bearer {ajeno}"}).status_code == 401


def test_auth_07_token_vencido_es_401_aunque_sea_superadmin(entorno):
    c, api = entorno
    vencido = jwt.encode({"sub": os.environ["API_USERNAME"], "role": "superadmin",
                          "exp": datetime.now(timezone.utc) - timedelta(seconds=1)},
                         api.JWT_SECRET, algorithm=api.JWT_ALGORITHM)
    assert c.get("/admin/users", headers={"Authorization": f"Bearer {vencido}"}).status_code == 401


def test_auth_08_viewer_valido_es_403_en_administracion(entorno):
    c, _ = entorno
    token = _login(c, os.environ["API_USERNAME_VIEWER"], os.environ["API_PASSWORD_VIEWER"]).json()["access_token"]
    assert c.get("/admin/users", headers={"Authorization": f"Bearer {token}"}).status_code == 403


def test_auth_09_superadmin_valido_lista_usuarios(entorno):
    c, _ = entorno
    r = c.get("/admin/users", headers={"Authorization": f"Bearer {_token_admin(c)}"})
    assert r.status_code == 200
    assert {u["username"] for u in r.json()} >= {os.environ["API_USERNAME"], os.environ["API_USERNAME_VIEWER"]}


# ── Clases de equivalencia del alta de usuarios ─────────────────────────────

@pytest.mark.parametrize("rol, esperado", [
    ("user", 201), ("admin", 201), ("superadmin", 201),   # clase válida
    ("root", 400), ("", 400),                              # clase inválida
])
def test_auth_10_rol_valido_o_invalido_al_crear(entorno, rol, esperado):
    c, _ = entorno
    admin = {"Authorization": f"Bearer {_token_admin(c)}"}
    r = c.post("/admin/users", json={"username": f"cp-auth-10-{rol or 'vacio'}", "password": "x", "role": rol}, headers=admin)
    assert r.status_code == esperado


def test_auth_11_usuario_duplicado_es_409(entorno):
    c, _ = entorno
    admin = {"Authorization": f"Bearer {_token_admin(c)}"}
    assert c.post("/admin/users", json={"username": "cp-auth-11", "password": "x"}, headers=admin).status_code == 201
    assert c.post("/admin/users", json={"username": "cp-auth-11", "password": "x"}, headers=admin).status_code == 409


def test_auth_12_el_superadmin_no_puede_desactivarse_a_si_mismo(entorno):
    c, _ = entorno
    admin = {"Authorization": f"Bearer {_token_admin(c)}"}
    r = c.patch(f"/admin/users/{os.environ['API_USERNAME']}", json={"active": False}, headers=admin)
    assert r.status_code == 400
