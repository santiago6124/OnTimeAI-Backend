// Lo común a los tres escenarios de k6: login, el mix de endpoints, las
// métricas por endpoint y el resumen que se escribe al terminar.
//
// Todo se parametriza por entorno:
//   BASE_URL      → la API bajo prueba (default http://localhost:8000). Nunca
//                   la de producción: ver docs/STRESS_K6.md.
//   API_USERNAME  → credenciales de prueba, las mismas que define el workflow
//   API_PASSWORD    (o tests/conftest.py en local). No son secretos.
//   SUMMARY_PATH  → dónde dejar el JSON del resumen (default
//                   k6/resultados/<escenario>-summary.json).
import http from "k6/http";
import { check } from "k6";
import { Trend } from "k6/metrics";

export const BASE_URL = (__ENV.BASE_URL || "http://localhost:8000").replace(/\/+$/, "");
const USUARIO = __ENV.API_USERNAME || "";
const CLAVE = __ENV.API_PASSWORD || "";

// Percentiles que k6 incluye en el resumen de cada Trend. p(99) no viene por
// defecto y es el que más dice sobre la cola.
export const TREND_STATS = ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max"];

// Umbrales comunes. El p95 lo fija cada escenario a partir de la PoC.
export function umbrales(p95MaxMs) {
  const t = {
    http_req_failed: ["rate<0.01"], // menos de 1 % de respuestas con error
    checks: ["rate>0.99"],          // al menos 99 % de los checks pasan
  };
  if (p95MaxMs) t.http_req_duration = [`p(95)<${p95MaxMs}`];
  return t;
}

// Una Trend por endpoint, para leer p95 por endpoint en el resumen sin
// tener que declarar un umbral sobre cada sub-métrica.
export const ENDPOINTS = [
  "login", "flights", "flight_detail", "flight_history", "cascade",
  "metrics_summary", "metrics_hourly", "metrics_model", "weather_atl",
];
const latencias = Object.fromEntries(
  ENDPOINTS.map((n) => [n, new Trend(`latencia_${n}`, true)]),
);

function cuerpoJson(res) {
  try {
    return res.json();
  } catch (_) {
    return undefined;
  }
}

function registrar(nombre, res) {
  latencias[nombre].add(res.timings.duration);
  const cuerpo = cuerpoJson(res);
  check(res, {
    [`${nombre}: 200`]: (r) => r.status === 200,
    [`${nombre}: JSON`]: () => cuerpo !== undefined,
  });
  return cuerpo;
}

// Login con las credenciales del entorno. Devuelve el token o null.
export function login() {
  const res = http.post(
    `${BASE_URL}/auth/login`,
    JSON.stringify({ username: USUARIO, password: CLAVE }),
    {
      headers: { "Content-Type": "application/json" },
      tags: { endpoint: "login", name: "POST /auth/login" },
    },
  );
  const cuerpo = registrar("login", res);
  return cuerpo && cuerpo.access_token ? cuerpo.access_token : null;
}

// GET autenticado. `plantilla` es la ruta sin el id, para que k6 no abra una
// serie por cada vuelo distinto.
export function pedir(nombre, ruta, token, plantilla) {
  const res = http.get(`${BASE_URL}${ruta}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    tags: { endpoint: nombre, name: `GET ${plantilla || ruta}` },
  });
  return registrar(nombre, res);
}

function elegirId(vuelos) {
  if (!Array.isArray(vuelos) || vuelos.length === 0) return null;
  return vuelos[Math.floor(Math.random() * vuelos.length)].fa_flight_id;
}

// El mix de un usuario del dashboard: la lista, el detalle de un vuelo al
// azar (con historial y cascada), las métricas y el clima. Nueve requests.
export function iteracionMix(token) {
  const vuelos = pedir("flights", "/flights", token);
  const id = elegirId(vuelos);
  check(vuelos, { "flights: hay vuelos de hoy": () => id !== null });
  if (id) {
    const codificado = encodeURIComponent(id);
    pedir("flight_detail", `/flights/${codificado}`, token, "/flights/{id}");
    pedir("flight_history", `/flight-history/${codificado}`, token, "/flight-history/{id}");
    pedir("cascade", `/cascade/${codificado}`, token, "/cascade/{id}");
  }
  pedir("metrics_summary", "/metrics/summary", token);
  pedir("metrics_hourly", "/metrics/hourly", token);
  pedir("metrics_model", "/metrics/model", token);
  pedir("weather_atl", "/weather/ATL", token);
}

// ── resumen ──────────────────────────────────────────────────────────────

function valor(data, metrica, campo) {
  const m = data.metrics[metrica];
  return m && m.values ? m.values[campo] : undefined;
}

function ms(v) {
  return v === undefined || v === null ? "—" : `${Math.round(v)} ms`;
}

function pct(v) {
  return v === undefined || v === null ? "—" : `${(100 * v).toFixed(2)} %`;
}

function lineasDeUmbrales(data) {
  const lineas = [];
  for (const [nombre, m] of Object.entries(data.metrics)) {
    for (const [expr, r] of Object.entries(m.thresholds || {})) {
      lineas.push(`  ${r.ok ? "OK   " : "FALLA"} ${nombre} ${expr}`);
    }
  }
  return lineas.length ? lineas : ["  (sin umbrales)"];
}

export function resumenTexto(escenario, data, extra) {
  const dur = (data.state.testRunDurationMs / 1000).toFixed(1);
  const lineas = [
    "",
    `== OnTimeAI · k6 · ${escenario} ==`,
    `duración ${dur} s · VUs máx ${valor(data, "vus_max", "max") ?? valor(data, "vus_max", "value") ?? "—"}` +
      ` · iteraciones ${valor(data, "iterations", "count") ?? 0}` +
      ` · requests ${valor(data, "http_reqs", "count") ?? 0} (${(valor(data, "http_reqs", "rate") ?? 0).toFixed(1)} req/s)`,
    `http_req_duration  p50 ${ms(valor(data, "http_req_duration", "med"))}` +
      ` · p95 ${ms(valor(data, "http_req_duration", "p(95)"))}` +
      ` · p99 ${ms(valor(data, "http_req_duration", "p(99)"))}` +
      ` · máx ${ms(valor(data, "http_req_duration", "max"))}`,
    `errores HTTP ${pct(valor(data, "http_req_failed", "rate"))} · checks ${pct(valor(data, "checks", "rate"))}`,
    "por endpoint (p95):",
    ...ENDPOINTS.map((n) => `  ${n.padEnd(16)} ${ms(valor(data, `latencia_${n}`, "p(95)"))}`),
    ...(extra || []),
    "umbrales:",
    ...lineasDeUmbrales(data),
    "",
  ];
  return lineas.join("\n");
}

// handleSummary de los tres escenarios: imprime el texto y guarda el JSON
// completo —es lo que lee scripts/k6_resumen.py—.
export function resumen(escenario, data, extra) {
  const ruta = __ENV.SUMMARY_PATH || `k6/resultados/${escenario}-summary.json`;
  data.ontimeai = {
    escenario,
    base_url: BASE_URL,
    generado_utc: new Date().toISOString(),
    k6: typeof __ENV.K6_VERSION === "string" ? __ENV.K6_VERSION : undefined,
  };
  return {
    stdout: resumenTexto(escenario, data, extra),
    [ruta]: JSON.stringify(data, null, 2),
  };
}
