#!/usr/bin/env bash
# Prueba de humo de la API desplegada. Lee BASE_URL, VIEWER_USER y VIEWER_PASS
# del entorno; escribe smoke-resumen.md y sale con 1 si algún check falla.
#
# Cada check tiene el criterio escrito antes de correr: código HTTP esperado y,
# cuando aplica, una condición sobre el JSON. Los checks son de sólo lectura.
set -u

BASE_URL=${BASE_URL:?falta BASE_URL}
fallos=0
: > smoke-resumen.md
echo "| Check | Esperado | Obtenido | Resultado |" >> smoke-resumen.md
echo "|---|---|---|---|" >> smoke-resumen.md

registrar() {  # nombre esperado obtenido ok(0/1)
  local marca="✅"; [ "$4" -eq 0 ] || { marca="❌"; fallos=$((fallos + 1)); }
  printf '| %s | %s | %s | %s |\n' "$1" "$2" "$3" "$marca" >> smoke-resumen.md
  printf '%s  %-45s esperado=%s obtenido=%s\n' "$marca" "$1" "$2" "$3"
}

# 1. El servicio responde y sirve la documentación (lo que pega el keepalive).
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 "$BASE_URL/docs")
registrar "GET /docs" 200 "$code" $([ "$code" = 200 ]; echo $?)

# 2. La autenticación rechaza credenciales inválidas (la tabla users existe y se consulta).
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 -H 'Content-Type: application/json' \
         -d '{"username":"nadie","password":"incorrecta"}' "$BASE_URL/auth/login")
registrar "POST /auth/login credenciales inválidas" 401 "$code" $([ "$code" = 401 ]; echo $?)

# 3. El viewer entra y recibe un token.
resp=$(curl -s --max-time 30 -w '\n%{http_code}' -H 'Content-Type: application/json' \
         -d "{\"username\":\"${VIEWER_USER:-viewer}\",\"password\":\"${VIEWER_PASS:-}\"}" "$BASE_URL/auth/login")
code=${resp##*$'\n'}; body=${resp%$'\n'*}
token=$(printf '%s' "$body" | jq -r '.access_token // empty' 2>/dev/null)
registrar "POST /auth/login viewer" "200 con access_token" "$code$([ -n "$token" ] && echo ' con token' || echo ' sin token')" $([ "$code" = 200 ] && [ -n "$token" ]; echo $?)
AUTH=(-H "Authorization: Bearer $token")

# 4. Las rutas que abre el dashboard al cargar.
check_json() {  # nombre ruta filtro-jq descripción
  local resp code body ok
  resp=$(curl -s --max-time 60 -w '\n%{http_code}' "${AUTH[@]}" "$BASE_URL$2")
  code=${resp##*$'\n'}; body=${resp%$'\n'*}
  ok=$(printf '%s' "$body" | jq -e "$3" >/dev/null 2>&1 && echo 0 || echo 1)
  registrar "$1" "200 y $4" "$code$([ "$ok" -eq 0 ] && echo ', ok' || echo ', JSON no cumple')" $([ "$code" = 200 ] && [ "$ok" -eq 0 ]; echo $?)
}
check_json "GET /metrics/summary" "/metrics/summary" 'has("total_flights") or has("total") or (type=="object" and length>0)' "objeto con métricas"
check_json "GET /flights" "/flights" 'type=="array"' "lista"
check_json "GET /metrics/model" "/metrics/model" 'type=="object" and length>0' "objeto con métricas del modelo"
check_json "GET /metrics/hourly" "/metrics/hourly" 'type=="array"' "lista"

# 5. Un id inexistente devuelve 404 con cuerpo JSON (la lógica de la app corre, no es un 404 del proxy).
#    Con token: /cascade exige autenticación en producción (sin token devuelve 401,
#    que fue lo que encontró el primer run del smoke y corrigió este check).
resp=$(curl -s --max-time 30 -w '\n%{http_code}' "${AUTH[@]}" "$BASE_URL/cascade/NO-EXISTE-0000")
code=${resp##*$'\n'}; body=${resp%$'\n'*}
es_json=$(printf '%s' "$body" | jq -e 'type=="object"' >/dev/null 2>&1 && echo 0 || echo 1)
registrar "GET /cascade/{id inexistente}" "404 JSON" "$code$([ "$es_json" -eq 0 ] && echo ' JSON' || echo ' no JSON')" $([ "$code" = 404 ] && [ "$es_json" -eq 0 ]; echo $?)

echo "" >> smoke-resumen.md
if [ "$fallos" -gt 0 ]; then
  echo "**$fallos check(s) fallaron.** Volver a la revisión anterior: \`gcloud run services update-traffic ontimeai-backend --to-revisions <anterior>=100\`." >> smoke-resumen.md
  echo "SMOKE FALLÓ: $fallos check(s)"; exit 1
fi
echo "Todos los checks pasaron." >> smoke-resumen.md
echo "SMOKE OK"
