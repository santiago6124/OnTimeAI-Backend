// Smoke: 1 usuario virtual, 30 segundos. Comprueba que el login funciona y que
// cada endpoint del mix responde 200 con JSON. No mide capacidad: es la
// condición previa para que el stress y el soak midan algo real.
//
//   k6 run -e API_USERNAME=... -e API_PASSWORD=... k6/smoke.js
import { sleep } from "k6";
import { login, iteracionMix, resumen, umbrales, TREND_STATS } from "./lib.js";

export const options = {
  vus: 1,
  duration: __ENV.SMOKE_DURATION || "30s",
  thresholds: umbrales(null),
  summaryTrendStats: TREND_STATS,
};

export default function () {
  // En el smoke el login se repite en cada iteración a propósito: también es
  // un endpoint que tiene que funcionar.
  const token = login();
  if (token) iteracionMix(token);
  sleep(1);
}

export function handleSummary(data) {
  return resumen("smoke", data);
}
