# Smoke post-deploy

Después de cada `Deploy` exitoso, `smoke.yml` verifica en menos de un minuto
que la revisión nueva de Cloud Run sirve lo que el Frontend y las apps piden al
abrir. También se puede disparar a mano (`workflow_dispatch`) desde cualquier
rama, por ejemplo para probar el propio smoke antes de mergearlo.

## Qué verifica (`scripts/smoke.sh`)

| Check | Criterio escrito antes de correr | Qué cubre |
|---|---|---|
| `GET /docs` | 200 | el contenedor arrancó y FastAPI responde |
| `POST /auth/login` con credenciales inválidas | 401 | la tabla `users` existe y se consulta |
| `POST /auth/login` con el viewer | 200 con `access_token` | los secretos del servicio están montados y el JWT se firma |
| `GET /metrics/summary`, `/flights`, `/metrics/model`, `/metrics/hourly` | 200 con la forma esperada | la base se descargó de GCS y las consultas corren |
| `GET /cascade/{id inexistente}` | 404 con cuerpo JSON | el 404 lo produce la app, no el proxy |

Todos los checks son de sólo lectura y usan el usuario viewer, cuya contraseña
se lee de Secret Manager en el job y se enmascara en el log.

## Qué hacer si falla

El deploy ya está sirviendo tráfico. Volver a la revisión anterior:

```bash
gcloud run revisions list --service ontimeai-backend --region us-central1 --project ontimeai-prod
gcloud run services update-traffic ontimeai-backend --region us-central1 --project ontimeai-prod \
  --to-revisions <revisión-anterior>=100
```

Después, abrir un defecto con el formulario del repo citando el run del smoke.

## Por qué no está dentro de `deploy.yml`

`workflow_run` deja el smoke en un workflow propio: se puede probar desde una
rama sin desplegar nada, y el deploy no cambia. El costo es que el smoke sólo
se dispara desde la versión de `smoke.yml` que está en `main`.
