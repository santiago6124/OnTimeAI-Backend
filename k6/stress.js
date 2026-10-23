// Stress: rampa de 0 a 50 usuarios virtuales en 3 minutos (1 m hasta 20,
// 1 m hasta 50, 1 m de vuelta a 0). Cada VU hace login una vez y después
// repite el mix del dashboard con un segundo de pausa entre iteraciones.
//
// Busca el punto en que la API, con un solo proceso uvicorn —como la única
// instancia de Cloud Run—, empieza a degradar. Lo que tiene que sostener bajo
// esa carga es no fallar: nada de 5xx ni de respuestas rotas. La latencia sí
// se degrada —la API sirve ~6,5 req/s de este mix y 50 VUs la saturan— y el
// umbral de p95 es un techo a ~2× lo medido en la PoC local, no un objetivo
// de servicio. Ver docs/STRESS_K6.md.
//
//   k6 run -e API_USERNAME=... -e API_PASSWORD=... k6/stress.js
import { sleep } from "k6";
import exec from "k6/execution";
import { login, iteracionMix, resumen, umbrales, TREND_STATS } from "./lib.js";

// p95 global de http_req_duration (ms). PoC local 2026-10-23: 14.611 ms.
const P95_MAX_MS = Number(__ENV.P95_MAX_MS || 30000);

export const options = {
  scenarios: {
    rampa: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: __ENV.STAGE_DURATION || "1m", target: 20 },
        { duration: __ENV.STAGE_DURATION || "1m", target: 50 },
        { duration: __ENV.STAGE_DURATION || "1m", target: 0 },
      ],
      gracefulRampDown: "15s",
    },
  },
  thresholds: umbrales(P95_MAX_MS),
  summaryTrendStats: TREND_STATS,
};

// Cada VU tiene su propio contexto de JS, así que esta variable es por VU:
// el login se hace una vez por usuario virtual, no por iteración.
let token = null;

export default function () {
  if (!token) {
    token = login();
    if (!token) exec.test.abort("el login falló: sin token no hay nada que medir");
  }
  iteracionMix(token);
  sleep(1);
}

export function handleSummary(data) {
  return resumen("stress", data, [`umbral p95: ${P95_MAX_MS} ms`]);
}
