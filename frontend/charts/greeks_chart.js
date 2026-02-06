export function drawGreeksChart(canvas, greeks) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const labels = ["Delta", "Gamma", "Vega", "Theta", "Rho"];
  const values = [greeks.delta, greeks.gamma, greeks.vega, greeks.theta, greeks.rho];
  const maxVal = Math.max(...values.map((v) => Math.abs(v))) || 1;
  const barWidth = canvas.width / labels.length;
  labels.forEach((label, idx) => {
    const value = values[idx];
    const barHeight = (Math.abs(value) / maxVal) * (canvas.height - 20);
    ctx.fillStyle = value >= 0 ? "#3dd598" : "#ff6b6b";
    ctx.fillRect(idx * barWidth + 10, canvas.height - barHeight - 10, barWidth - 20, barHeight);
    ctx.fillStyle = "#f5f6fa";
    ctx.fillText(label, idx * barWidth + 10, canvas.height - 2);
  });
}
