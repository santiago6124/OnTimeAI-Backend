// Soak (esfuerzo sostenido): 20 usuarios virtuales constantes durante 10
// minutos con el mismo mix que el stress. La pregunta no es cuánto aguanta
// sino si se degrada con el tiempo: fugas de memoria, conexiones SQLite que
// no se cierran, caches que crecen. Por eso el resumen compara el p95 de la
// iteración completa en el primer minuto con el del último.
//
// 20 VUs están muy por encima de la rodilla de la API (~6,5 req/s, 1-2 VUs de
// este mix): la latencia es alta pero estable, y lo que se vigila es que no
// crezca con el tiempo ni aparezcan errores. El umbral de p95 es un techo a
// ~2× lo medido en la PoC local, no un objetivo de servicio. Ver
// docs/STRESS_K6.md.
//
//   k6 run -e API_USERNAME=... -e API_PASSWORD=... k6/soak.js
//   k6 run -e SOAK_DURATION=3m ... k6/soak.js      # versión corta para una PoC
import { sleep } from "k6";
import exec from "k6/execution";
import { Trend } from "k6/metrics";
import { login, iteracionMix, resumen, umbrales, TREND_STATS } from "./lib.js";

const DURACION = __ENV.SOAK_DURATION || "10m";
const VUS = Number(__ENV.SOAK_VUS || 20);
// p95 global de http_req_duration (ms). PoC local 2026-10-23: 17.461 ms.
const P95_MAX_MS = Number(__ENV.P95_MAX_MS || 35000);

export const options = {
  scenarios: {
    sostenido: {
      executor: "constant-vus",
      vus: VUS,
      duration: DURACION,
    },
  },
  thresholds: umbrales(P95_MAX_MS),
  summaryTrendStats: TREND_STATS,
};

// Latencia de las iteraciones del primer y del último minuto, para ver la
// deriva. Se alimentan con la duración total de la iteración (todo el mix).
const primerMinuto = new Trend("iteracion_primer_minuto", true);
const ultimoMinuto = new Trend("iteracion_ultimo_minuto", true);

function aMs(duracion) {
  const m = /^(\d+(?:\.\d+)?)(ms|s|m|h)$/.exec(duracion.trim());
  if (!m) throw new Error(`SOAK_DURATION inválida: ${duracion}`);
  return Number(m[1]) * { ms: 1, s: 1000, m: 60000, h: 3600000 }[m[2]];
}
const DURACION_MS = aMs(DURACION);

let token = null;

export default function () {
  if (!token) {
    token = login();
    if (!token) exec.test.abort("el login falló: sin token no hay nada que medir");
  }
  const inicio = Date.now();
  iteracionMix(token);
  const tardo = Date.now() - inicio;

  const transcurrido = inicio - exec.scenario.startTime;
  if (transcurrido < 60000) primerMinuto.add(tardo);
  else if (transcurrido >= DURACION_MS - 60000) ultimoMinuto.add(tardo);
  sleep(1);
}

export function handleSummary(data) {
  const p = (n) => {
    const m = data.metrics[n];
    return m && m.values ? `${Math.round(m.values["p(95)"])} ms` : "—";
  };
  return resumen("soak", data, [
    `deriva (p95 de la iteración completa): primer minuto ${p("iteracion_primer_minuto")}` +
      ` · último minuto ${p("iteracion_ultimo_minuto")}`,
    `umbral p95: ${P95_MAX_MS} ms · duración ${DURACION} · ${VUS} VUs`,
  ]);
}
