# Secretos del backend

Los secretos viven en **Secret Manager** y se inyectan en Cloud Run con
`--set-secrets`. No hay valores por defecto en el código: si falta una variable,
el contenedor no arranca.

## Qué hay

| Secret Manager | Variable | Qué es |
|---|---|---|
| `jwt-secret-key` | `JWT_SECRET_KEY` | Clave de firma HS256 de los JWT |
| `api-password` | `API_PASSWORD` | Contraseña del usuario seed `admin` (superadmin) |
| `api-password-viewer` | `API_PASSWORD_VIEWER` | Contraseña del usuario seed `viewer` |
| `aeroapi-key` | `AEROAPI_KEY` | Clave de FlightAware, usada por el live job |

Los nombres de usuario (`API_USERNAME`, `API_USERNAME_VIEWER`) siguen como
variables de entorno planas: no son secretos.

## Leer un valor

```bash
gcloud secrets versions access latest --secret=api-password --project=ontimeai-prod
```

## Por qué no hay defaults

Antes el código tenía:

```python
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "ontimeai-dev-secret-change-in-prod-32chars")
```

Un default convierte una variable faltante en un arranque **exitoso** con una
credencial conocida. El servicio queda en pie, responde bien, y nada avisa. Con
el secreto JWT publicado en el repositorio, cualquiera que lo lea puede firmar un
token de rol `superadmin` y usar `/admin/users` para crear cuentas.

Ahora `_require_secret()` aborta el arranque. Cloud Run deja la revisión anterior
sirviendo y el error queda en los logs del despliegue, que es el modo de falla
correcto: ruidoso y sin degradar a un estado inseguro.

## Rotar

### El secreto JWT

```bash
python3 -c "import secrets; print(secrets.token_hex(32))" \
  | gcloud secrets versions add jwt-secret-key --project=ontimeai-prod --data-file=-
```

Después hay que redesplegar el backend para que tome la versión nueva.

**Efecto inmediato: se invalidan todas las sesiones activas.** Los tokens
firmados con la clave anterior dejan de validar y todos los usuarios tienen que
volver a iniciar sesión. No hay período de gracia.

### Una contraseña

Acá hay una trampa. Subir una versión nueva del secreto **no cambia la
contraseña de un usuario que ya existe**:

```python
if username and not con.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
    con.execute("INSERT INTO users ...")
```

El alta inicial es idempotente: sólo inserta si el usuario **no** existe. Y
`users.db` persiste en GCS (`gs://ontimeai-prod-live-db/users.db`) entre
despliegues, así que la fila sobrevive.

Las variables `API_PASSWORD` y `API_PASSWORD_VIEWER` sólo tienen efecto la
**primera vez**, cuando se crea la base de usuarios desde cero.

Para rotar de verdad hay dos caminos:

**1. Vía la API, con una sesión de superadmin:**

```bash
TOKEN=$(curl -s -X POST "$BACKEND/auth/login" \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"<la actual>"}' | jq -r .access_token)

curl -X PATCH "$BACKEND/admin/users/admin" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"password":"<la nueva>"}'
```

**2. Borrando `users.db` de GCS** para que el alta inicial vuelva a correr. Es
destructivo: se pierden todos los usuarios creados a mano y sus roles.

En cualquier caso, el valor en Secret Manager y el de la base tienen que quedar
iguales, o el alta inicial de un entorno nuevo va a crear un `admin` con una
contraseña distinta de la que está documentada.

## Entorno local

`api.py` carga `.env.local` con `load_dotenv()`. Ese archivo no está versionado.
Para trabajar en local hay que crearlo con las cuatro variables; los valores
pueden ser cualquier cosa mientras no se apunte a datos de producción.

Los tests no lo necesitan: `tests/conftest.py` define valores de prueba antes de
importar `api`. Eso es a propósito — un runner de CI no tiene `.env.local`, y sin
esa red los tests pasarían en local y fallarían en CI.

## Dar acceso a una service account

```bash
gcloud secrets add-iam-policy-binding jwt-secret-key \
  --member="serviceAccount:<sa>" \
  --role="roles/secretmanager.secretAccessor" \
  --project=ontimeai-prod
```

Hoy lo tienen:

- `871707213932-compute@developer.gserviceaccount.com` — runtime de Cloud Run
- `github-deployer@ontimeai-prod.iam.gserviceaccount.com` — despliegues desde
  GitHub Actions (ver `docs/GITHUB_GCP_AUTH.md`)
