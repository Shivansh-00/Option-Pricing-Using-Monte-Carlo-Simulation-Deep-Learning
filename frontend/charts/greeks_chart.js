function fnum(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

export function drawGreeksChart(canvas, greeks) {
  if (!canvas || typeof canvas.getContext !== "function") return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const g = greeks || {};
  const labels = ["Delta", "Gamma", "Vega", "Theta", "Rho"];
  const values = [fnum(g.delta), fnum(g.gamma), fnum(g.vega), fnum(g.theta), fnum(g.rho)];

  const width = canvas.width;
  const height = canvas.height;
  const margin = { top: 14, right: 10, bottom: 32, left: 10 };
  const chartW = width - margin.left - margin.right;
  const chartH = height - margin.top - margin.bottom;
  const baseline = margin.top + chartH / 2;

  ctx.clearRect(0, 0, width, height);
  ctx.font = "11px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "center";

  const maxAbs = Math.max(1, ...values.map((v) => Math.abs(v)));

  ctx.strokeStyle = "rgba(245, 246, 250, 0.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margin.left, baseline);
  ctx.lineTo(margin.left + chartW, baseline);
  ctx.stroke();

  const slotW = chartW / labels.length;
  const barW = Math.max(16, slotW - 18);
  labels.forEach((label, idx) => {
    const value = values[idx];
    const h = (Math.abs(value) / maxAbs) * (chartH / 2 - 6);
    const x = margin.left + slotW * idx + (slotW - barW) / 2;
    const y = value >= 0 ? baseline - h : baseline;

    ctx.fillStyle = value >= 0 ? "#3dd598" : "#ff6b6b";
    ctx.fillRect(x, y, barW, h);

    ctx.fillStyle = "#f5f6fa";
    ctx.fillText(label, x + barW / 2, height - 10);
  });
}
