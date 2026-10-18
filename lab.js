(() => {
  const canvas = document.getElementById('lab');
  const ctx = canvas.getContext('2d');
  const status = document.getElementById('status');
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const K = 3, CELL = 14;
  let pts = [], cls = 0, w = 0, h = 0, dpr = 1;

  function resize() {
    dpr = window.devicePixelRatio || 1;
    w = canvas.clientWidth; h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function predict(x, y) {
    const near = pts.map(p => ({ c: p.c, d: (p.x - x) ** 2 + (p.y - y) ** 2 }))
      .sort((a, b) => a.d - b.d).slice(0, K);
    const votes = [0, 0];
    near.forEach(n => votes[n.c] += 1 / (1 + n.d / 4000));
    return votes[0] === votes[1] ? -1 : votes[0] > votes[1] ? 0 : 1;
  }

  function draw() {
    const colors = [css('--pink'), css('--cyan')];
    ctx.clearRect(0, 0, w, h);
    if (pts.length) {
      ctx.globalAlpha = 0.22;
      for (let x = 0; x < w; x += CELL)
        for (let y = 0; y < h; y += CELL) {
          const c = predict(x + CELL / 2, y + CELL / 2);
          if (c < 0) continue;
          ctx.fillStyle = colors[c];
          ctx.fillRect(x, y, CELL, CELL);
        }
      ctx.globalAlpha = 1;
    }
    pts.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
      ctx.fillStyle = colors[p.c];
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = css('--bg');
      ctx.stroke();
    });
  }

  function say() {
    const a = pts.filter(p => p.c === 0).length, b = pts.length - a;
    if (!pts.length) status.textContent = 'Click to add your first point.';
    else if (!a || !b) status.textContent = 'Now add a few points of the other colour and watch the border appear.';
    else if (pts.length < 8) status.textContent = `${pts.length} points. The model is starting to get the idea.`;
    else status.textContent = `${pts.length} points. You're the teacher here. Try putting a pink one deep in cyan territory.`;
  }

  canvas.addEventListener('pointerdown', e => {
    const r = canvas.getBoundingClientRect();
    pts.push({ x: e.clientX - r.left, y: e.clientY - r.top, c: cls });
    draw(); say();
  });

  document.querySelectorAll('.tool[data-cls]').forEach(b => b.addEventListener('click', () => {
    cls = +b.dataset.cls;
    document.querySelectorAll('.tool[data-cls]').forEach(x => x.classList.toggle('active', x === b));
  }));

  document.getElementById('clear').addEventListener('click', () => { pts = []; draw(); say(); });

  document.getElementById('random').addEventListener('click', () => {
    pts = [];
    const blob = (cx, cy, c) => { for (let i = 0; i < 9; i++) pts.push({
      x: Math.min(w - 10, Math.max(10, cx + (Math.random() - .5) * w * .3)),
      y: Math.min(h - 10, Math.max(10, cy + (Math.random() - .5) * h * .5)), c }); };
    const flip = Math.random() < .5;
    blob(w * .28, h * .5, flip ? 0 : 1);
    blob(w * .72, h * .5, flip ? 1 : 0);
    draw(); say();
  });

  window.addEventListener('resize', resize);
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
  resize();
})();
