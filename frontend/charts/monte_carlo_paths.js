export function drawMonteCarloPaths(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#4c7fff";
  for (let p = 0; p < 10; p += 1) {
    ctx.beginPath();
    let value = 100 + Math.random() * 10;
    ctx.moveTo(0, canvas.height - value);
    for (let i = 1; i <= 30; i += 1) {
      value += (Math.random() - 0.5) * 8;
      ctx.lineTo((canvas.width / 30) * i, canvas.height - value);
    }
    ctx.stroke();
  }
}
