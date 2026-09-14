# Autenticación de GitHub Actions contra GCP

Los workflows se autentican con **Workload Identity Federation**: GitHub emite un
token OIDC por ejecución y GCP lo intercambia por credenciales de corta duración.
**No hay ninguna clave guardada en los secrets del repositorio.**

La alternativa —una clave JSON de service account en un secret— se descartó: es
una credencial de larga duración, no rota, y queda en el historial de quien la
copie. Si alguien la filtra, hay que rotarla a mano en los tres repos.

## Qué quedó montado

| Recurso | Valor |
|---|---|
| Proyecto | `ontimeai-prod` (número `871707213932`) |
| Pool | `github` |
| Provider | `github` |
| Service account | `github-deployer@ontimeai-prod.iam.gserviceaccount.com` |

Referencia completa del provider, la que va en los workflows:

```
projects/871707213932/locations/global/workloadIdentityPools/github/providers/github
```

## Permisos de la service account

Se creó una SA dedicada en vez de reutilizar la default de compute, que tiene
permisos amplios sobre todo el proyecto.

| Rol | Alcance | Para qué |
|---|---|---|
| `roles/run.admin` | proyecto | Desplegar servicios y jobs |
| `roles/cloudbuild.builds.editor` | proyecto | Disparar builds |
| `roles/artifactregistry.writer` | proyecto | Pushear imágenes |
| `roles/logging.viewer` | proyecto | Leer logs de build al fallar |
| `roles/iam.serviceAccountUser` | **solo** sobre la SA de compute | Actuar como la SA de runtime |
| `roles/storage.admin` | **solo** `gs://ontimeai-prod_cloudbuild` | Subir el contexto de build |

Los dos últimos están acotados a propósito: no hace falta que el deployer pueda
tocar `ontimeai-prod-live-db` ni `ontimeai-prod-training`.

`gcr.io` está redirigido a Artifact Registry en este proyecto, así que
`artifactregistry.writer` cubre los pushes a `gcr.io/ontimeai-prod/...`.

## Qué protege y qué no

**Restringido al owner.** El provider lleva
`--attribute-condition="assertion.repository_owner=='santiago6124'"`. **Esta
condición no es opcional**: sin ella cualquier repositorio de GitHub del mundo
puede pedir un token e impersonar la service account.

**Restringido a tres repos.** El rol `workloadIdentityUser` está bindeado sólo a
`OnTimeAI-Backend`, `OnTimeAI-Frontend` y `OnTimeAI-Scrapper`. Otro repo del
mismo owner no puede impersonar la SA.

**Lo que NO separa.** Los tres repos comparten la misma service account, así que
tienen los mismos permisos: el workflow del Frontend *podría* desplegar el
Backend. Para separarlos haría falta una SA por repo. Se dejó así a propósito —
un solo equipo, un solo proyecto GCP— pero conviene saberlo antes de sumar un
repo de terceros.

**No hay restricción por rama.** Un workflow en cualquier rama de esos repos
puede autenticar. Si se quiere limitar a `main`, se agrega al binding:

```
attribute.repository/santiago6124/OnTimeAI-Backend/attribute.ref/refs/heads/main
```

Requiere que `attribute.ref` esté en el mapping, que ya lo está.

## Cómo se usa en un workflow

```yaml
permissions:
  id-token: write        # sin esto el auth falla con "missing id-token permission"
  contents: read

steps:
  - uses: google-github-actions/auth@v2
    with:
      project_id: ontimeai-prod
      workload_identity_provider: projects/871707213932/locations/global/workloadIdentityPools/github/providers/github
      service_account: github-deployer@ontimeai-prod.iam.gserviceaccount.com

  - uses: google-github-actions/setup-gcloud@v2
```

`.github/workflows/gcp-auth-check.yml` lo verifica de punta a punta. Es
`workflow_dispatch`: se corre a mano desde la pestaña Actions.

## Cómo rehacerlo desde cero

Los créditos se agotaron una vez y hubo que migrar de `ontimeai` a
`ontimeai-prod`. Si vuelve a pasar, esto es lo que hay que repetir:

```bash
P=nuevo-proyecto
NUM=$(gcloud projects describe $P --format='value(projectNumber)')

gcloud services enable sts.googleapis.com iamcredentials.googleapis.com --project=$P

gcloud iam workload-identity-pools create github \
  --location=global --project=$P --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc github \
  --workload-identity-pool=github --location=global --project=$P \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner,attribute.ref=assertion.ref" \
  --attribute-condition="assertion.repository_owner=='santiago6124'"

gcloud iam service-accounts create github-deployer --project=$P

SA="github-deployer@$P.iam.gserviceaccount.com"
for R in roles/run.admin roles/cloudbuild.builds.editor \
         roles/artifactregistry.writer roles/logging.viewer; do
  gcloud projects add-iam-policy-binding $P --member="serviceAccount:$SA" --role="$R"
done

gcloud iam service-accounts add-iam-policy-binding "$NUM-compute@developer.gserviceaccount.com" \
  --member="serviceAccount:$SA" --role="roles/iam.serviceAccountUser" --project=$P

gcloud storage buckets add-iam-policy-binding gs://${P}_cloudbuild \
  --member="serviceAccount:$SA" --role="roles/storage.admin" --project=$P

POOL="principalSet://iam.googleapis.com/projects/$NUM/locations/global/workloadIdentityPools/github/attribute.repository"
for R in OnTimeAI-Backend OnTimeAI-Frontend OnTimeAI-Scrapper; do
  gcloud iam service-accounts add-iam-policy-binding "$SA" --project=$P \
    --role="roles/iam.workloadIdentityUser" --member="$POOL/santiago6124/$R"
done
```

Después hay que actualizar el número de proyecto en el
`workload_identity_provider` de cada workflow.

## Diagnóstico

| Error | Causa |
|---|---|
| `missing id-token permission` | Falta `permissions: id-token: write` en el workflow |
| `Unable to acquire impersonated credentials` | El repo no tiene binding `workloadIdentityUser` |
| `The attribute condition must reference one of the provider's claims` | El mapping no incluye el atributo que usa la condición |
| `Permission denied on resource project` | Falta un rol; revisar la tabla de permisos |
