// New-run form: POST /api/runs, then redirect to the new run detail page.
(() => {
  const form = document.getElementById('new-run-form');
  if (!form) return;
  const status = document.getElementById('new-run-status');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    const body = {
      brief: fd.get('brief'),
      max_iters: Number(fd.get('max_iters')),
      patience: Number(fd.get('patience')),
      tolerance: Number(fd.get('tolerance')),
      model: fd.get('model') || null,
    };
    const btn = form.querySelector('button');
    btn.disabled = true;
    status.textContent = 'starting…';
    try {
      const res = await fetch('/api/runs', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(await res.text());
      const { run_id } = await res.json();
      window.location.href = `/runs/${run_id}`;
    } catch (err) {
      status.textContent = 'error: ' + err.message;
      btn.disabled = false;
    }
  });
})();

// Run detail: render the score chart from rendered iteration cards, then
// open an SSE stream to append live updates while the run is running.
(() => {
  const runId = document.body.dataset.runId;
  if (!runId) return;

  const grid = document.getElementById('iter-grid');
  const chartEl = document.getElementById('score-chart');
  const statBest = document.getElementById('stat-best-score');
  const statBestIter = document.getElementById('stat-best-iter');
  const statCount = document.getElementById('stat-iter-count');

  // Pull seed data out of the already-rendered iteration cards.
  function readIters() {
    return Array.from(grid.querySelectorAll('.iter-card')).map((card) => {
      const num = Number(card.dataset.iter);
      const composite = Number(card.querySelector('.badge').textContent);
      const sus = Number(card.querySelector('.stats b').textContent);
      return { iter: num, composite, sus };
    });
  }

  // Simple SVG line chart: composite (solid blue) + SUS (dashed gray) over iterations.
  function renderChart(iters) {
    if (!iters.length) {
      chartEl.innerHTML = '';
      return;
    }
    const W = 800, H = 220, padL = 32, padR = 12, padT = 16, padB = 24;
    const maxIter = Math.max(iters.length, 5);
    const x = (i) => padL + ((i - 1) / Math.max(1, maxIter - 1)) * (W - padL - padR);
    const y = (v) => padT + (1 - v / 100) * (H - padT - padB);

    const gridLines = [0, 25, 50, 75, 100].map((v) =>
      `<line class="grid-line" x1="${padL}" y1="${y(v)}" x2="${W - padR}" y2="${y(v)}" />
       <text class="axis-label" x="4" y="${y(v) + 3}">${v}</text>`
    ).join('');

    const pointsCompo = iters.map((it) => `${x(it.iter)},${y(it.composite)}`).join(' ');
    const pointsSus = iters.map((it) => `${x(it.iter)},${y(it.sus)}`).join(' ');
    const dots = iters.map((it) =>
      `<circle class="point" cx="${x(it.iter)}" cy="${y(it.composite)}" r="3" />`
    ).join('');

    chartEl.innerHTML = `
      ${gridLines}
      <polyline class="line-sus" points="${pointsSus}" />
      <polyline class="line-composite" points="${pointsCompo}" />
      ${dots}
    `;
  }

  function iterCardHtml(it) {
    const score = it.composite_score;
    const cls = score >= 80 ? 'score-good' : score >= 60 ? 'score-ok' : 'score-bad';
    const suggestions = (it.suggestions || []).map((s) =>
      `<li>${escapeHtml(s)}</li>`).join('');
    return `<article class="iter-card appearing" data-iter="${it.iter}">
      <header>
        <span class="iter-num">#${it.iter}</span>
        <span class="badge ${cls}">${score.toFixed(0)}</span>
      </header>
      <a href="/runs/${runId}/iters/${it.iter}/site" target="_blank" class="thumb">
        <img src="/runs/${runId}/iters/${it.iter}/screenshot" alt="Iter ${it.iter}" />
      </a>
      <div class="stats">
        <span>SUS <b>${it.sus_score.toFixed(0)}</b></span>
        <span>a11y penalty <b>${it.axe_penalty.toFixed(0)}</b></span>
      </div>
      <p class="feedback">${escapeHtml(it.feedback)}</p>
      <details>
        <summary>Suggestions</summary>
        <ul>${suggestions}</ul>
      </details>
    </article>`;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  }

  function updateSummary(iters) {
    if (!iters.length) return;
    let bestIter = iters[0], best = iters[0].composite;
    for (const it of iters) {
      if (it.composite > best) {
        best = it.composite;
        bestIter = it;
      }
    }
    statBest.textContent = best.toFixed(0);
    statBest.className = best >= 80 ? 'score-good' : best >= 60 ? 'score-ok' : 'score-bad';
    statBestIter.textContent = bestIter.iter;
    statCount.textContent = iters.length;
  }

  // Initial render from server-rendered cards.
  let iters = readIters();
  renderChart(iters);

  if (document.body.dataset.running !== '1') return;

  // Live updates via SSE — tell the server where we already are.
  const since = iters.length ? iters[iters.length - 1].iter : 0;
  const es = new EventSource(`/runs/${runId}/stream?since=${since}`);
  es.addEventListener('iteration', (e) => {
    const payload = JSON.parse(e.data);
    const it = payload.iter;
    // Append card
    grid.insertAdjacentHTML('beforeend', iterCardHtml(it));
    // Update chart data
    iters.push({ iter: it.iter, composite: it.composite_score, sus: it.sus_score });
    renderChart(iters);
    updateSummary(iters);
  });
  es.addEventListener('done', (e) => {
    const { run } = JSON.parse(e.data);
    const badge = document.querySelector('h1 .status');
    if (badge) {
      badge.className = `status status-${run.status}`;
      badge.textContent = run.status;
    }
    es.close();
  });
  es.addEventListener('error', () => {
    // Connection drop; let the browser auto-reconnect unless the run is done.
  });
})();
