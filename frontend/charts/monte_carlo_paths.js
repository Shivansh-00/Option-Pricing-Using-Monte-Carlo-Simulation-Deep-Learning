function finite(n, fallback = 0) {
  const v = Number(n);
  return Number.isFinite(v) ? v : fallback;
}

function generateFallbackPaths(pathCount = 10, steps = 30) {
  const paths = [];
  for (let p = 0; p < pathCount; p += 1) {
    const series = [];
    let value = 100 + Math.random() * 10;
    for (let i = 0; i <= steps; i += 1) {
      if (i > 0) value += (Math.random() - 0.5) * 8;
      series.push(value);
    }
    paths.push(series);
  }
  return paths;
}

export function drawMonteCarloPaths(canvas, mcData = null) {
  if (!canvas || typeof canvas.getContext !== "function") return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const width = canvas.width;
  const height = canvas.height;
  const margin = { top: 12, right: 12, bottom: 20, left: 12 };
  const chartW = width - margin.left - margin.right;
  const chartH = height - margin.top - margin.bottom;

  const paths = Array.isArray(mcData?.paths) && mcData.paths.length
    ? mcData.paths
    : generateFallbackPaths();

  let minY = Infinity;
  let maxY = -Infinity;
  paths.forEach((path) => {
    path.forEach((v) => {
      const n = finite(v, 100);
      if (n < minY) minY = n;
      if (n > maxY) maxY = n;
    });
  });
  if (!Number.isFinite(minY) || !Number.isFinite(maxY) || maxY <= minY) {
    minY = 80;
    maxY = 120;
  }

  const yPad = (maxY - minY) * 0.08;
  minY -= yPad;
  maxY += yPad;
  const toY = (v) => margin.top + chartH - ((finite(v, minY) - minY) / (maxY - minY)) * chartH;

  ctx.clearRect(0, 0, width, height);

  if (Array.isArray(mcData?.mean_path) && Array.isArray(mcData?.ci_lower) && Array.isArray(mcData?.ci_upper)) {
    const n = Math.min(mcData.mean_path.length, mcData.ci_lower.length, mcData.ci_upper.length);
    if (n > 1) {
      ctx.beginPath();
      for (let i = 0; i < n; i += 1) {
        const x = margin.left + (chartW * i) / (n - 1);
        const y = toY(mcData.ci_upper[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      for (let i = n - 1; i >= 0; i -= 1) {
        const x = margin.left + (chartW * i) / (n - 1);
        const y = toY(mcData.ci_lower[i]);
        ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = "rgba(76, 127, 255, 0.15)";
      ctx.fill();
    }
  }

  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(76, 127, 255, 0.35)";
  paths.slice(0, 20).forEach((path) => {
    if (!Array.isArray(path) || path.length < 2) return;
    const denom = Math.max(1, path.length - 1);
    ctx.beginPath();
    for (let i = 0; i < path.length; i += 1) {
      const x = margin.left + (chartW * i) / denom;
      const y = toY(path[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });

  if (Array.isArray(mcData?.mean_path) && mcData.mean_path.length > 1) {
    const denom = mcData.mean_path.length - 1;
    ctx.beginPath();
    ctx.strokeStyle = "#f4b860";
    ctx.lineWidth = 2;
    mcData.mean_path.forEach((v, i) => {
      const x = margin.left + (chartW * i) / denom;
      const y = toY(v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}
