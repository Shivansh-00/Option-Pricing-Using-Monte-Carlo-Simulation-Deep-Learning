import { drawComparisonChart } from "./charts/comparison_chart.js";
import { drawGreeksChart } from "./charts/greeks_chart.js";
import { drawMonteCarloPaths } from "./charts/monte_carlo_paths.js";

const apiBase = "";

function getPayload() {
  return {
    spot: Number(document.getElementById("spot").value),
    strike: Number(document.getElementById("strike").value),
    maturity: Number(document.getElementById("maturity").value),
    rate: Number(document.getElementById("rate").value),
    volatility: Number(document.getElementById("vol").value),
    option_type: document.getElementById("optionType").value,
    steps: 252,
    paths: 10000,
  };
}

async function postJson(path, payload) {
  const response = await fetch(`${apiBase}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

async function runPricing() {
  try {
    const payload = getPayload();
    const [bs, mc, dl, greeks, iv] = await Promise.all([
      postJson("/api/v1/pricing/bs", payload),
      postJson("/api/v1/pricing/mc", payload),
      postJson("/api/v1/dl/forecast", payload),
      postJson("/api/v1/pricing/greeks", payload),
      postJson("/api/v1/ml/iv-predict", {
        spot: payload.spot,
        rate: payload.rate,
        maturity: payload.maturity,
        realized_vol: payload.volatility,
        vix: payload.volatility * 1.1,
        skew: 0.1,
      }),
    ]);

    document.getElementById("bsPrice").textContent = bs.price.toFixed(4);
    document.getElementById("mcPrice").textContent = mc.price.toFixed(4);
    document.getElementById("dlPrice").textContent = dl.forecast_price.toFixed(4);
    document.getElementById("delta").textContent = greeks.delta.toFixed(4);
    document.getElementById("gamma").textContent = greeks.gamma.toFixed(4);
    document.getElementById("vega").textContent = greeks.vega.toFixed(4);
    document.getElementById("theta").textContent = greeks.theta.toFixed(4);
    document.getElementById("rho").textContent = greeks.rho.toFixed(4);
    document.getElementById("regime").textContent = iv.regime;

    drawComparisonChart(
      document.getElementById("comparisonChart"),
      bs.price,
      mc.price,
      dl.forecast_price,
    );
    drawGreeksChart(document.getElementById("greeksChart"), greeks);
    drawMonteCarloPaths(document.getElementById("mcPaths"));
  } catch (error) {
    alert(error.message);
  }
}

async function askExplain() {
  try {
    const question = document.getElementById("question").value || "Explain pricing";
    const response = await postJson("/api/v1/ai/explain", {
      question,
      context: { model: "hybrid", focus: "volatility regime" },
    });
    document.getElementById("explainAnswer").textContent = response.answer;
  } catch (error) {
    alert(error.message);
  }
}

document.getElementById("runPricing").addEventListener("click", runPricing);
document.getElementById("askExplain").addEventListener("click", askExplain);

runPricing();
