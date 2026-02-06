export function drawComparisonChart(canvas, bs, mc, dl) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const labels = ["BS", "MC", "DL"];
  const values = [bs, mc, dl];
  const maxVal = Math.max(...values, 1);
  const barWidth = canvas.width / labels.length;
  labels.forEach((label, idx) => {
    const value = values[idx];
    const barHeight = (value / maxVal) * (canvas.height - 20);
    ctx.fillStyle = ["#4c7fff", "#f4b860", "#8b5cf6"][idx];
    ctx.fillRect(idx * barWidth + 10, canvas.height - barHeight - 10, barWidth - 20, barHeight);
    ctx.fillStyle = "#f5f6fa";
    ctx.fillText(label, idx * barWidth + 20, canvas.height - 2);
  });
}
