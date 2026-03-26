function toFinite(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

export function drawComparisonChart(canvas, bs, mc, dl) {
  if (!canvas || typeof canvas.getContext !== "function") return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const labels = ["BS", "MC", "DL"];
  const values = [toFinite(bs), toFinite(mc), toFinite(dl)];
  const palette = ["#4c7fff", "#f4b860", "#2ec27e"];

  const width = canvas.width;
  const height = canvas.height;
  const margin = { top: 16, right: 12, bottom: 34, left: 10 };
  const chartW = width - margin.left - margin.right;
  const chartH = height - margin.top - margin.bottom;

  ctx.clearRect(0, 0, width, height);
  ctx.font = "12px system-ui, -apple-system, sans-serif";
  ctx.textBaseline = "middle";

  const maxVal = Math.max(1, ...values.map((v) => Math.abs(v)));

  // Grid lines for quick visual comparison.
  ctx.strokeStyle = "rgba(245, 246, 250, 0.16)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = margin.top + (chartH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + chartW, y);
    ctx.stroke();
  }

  const slotW = chartW / labels.length;
  const barW = Math.max(18, slotW - 22);
  labels.forEach((label, idx) => {
    const v = values[idx];
    const x = margin.left + slotW * idx + (slotW - barW) / 2;
    const h = (Math.abs(v) / maxVal) * chartH;
    const y = margin.top + chartH - h;

    const grad = ctx.createLinearGradient(0, y, 0, margin.top + chartH);
    grad.addColorStop(0, palette[idx]);
    grad.addColorStop(1, "rgba(255, 255, 255, 0.2)");
    ctx.fillStyle = grad;
    ctx.fillRect(x, y, barW, h);

    ctx.fillStyle = "#f5f6fa";
    ctx.textAlign = "center";
    ctx.fillText(label, x + barW / 2, height - 12);
    ctx.fillText(v.toFixed(2), x + barW / 2, Math.max(margin.top + 8, y - 8));
  });
}
