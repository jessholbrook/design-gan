// New-run form: POST /api/runs, then redirect to the new run detail page.
// When the server gates runs with DESIGN_GAN_START_TOKEN, the form carries
// a token field; we cache the value in localStorage so returning visitors
// don't have to paste it every time. The kind selector toggles the brief
// placeholder text and reveals conversation-only fields.
(() => {
  const form = document.getElementById('new-run-form');
  if (!form) return;
  const status = document.getElementById('new-run-status');
  const TOKEN_KEY = 'design_gan_start_token';
  const tokenInput = form.querySelector('input[name="token"]');
  if (tokenInput) {
    const cached = localStorage.getItem(TOKEN_KEY);
    if (cached) tokenInput.value = cached;
  }

  // Kind selector: toggle conversation-only fields + brief placeholder.
  const kindSel = form.querySelector('select[name="kind"]');
  const briefTA = form.querySelector('textarea[name="brief"]');
  const briefLabel = form.querySelector('[data-brief-label]');
  const conversationOnly = form.querySelectorAll('[data-conversation-only]');
  const designOnly = form.querySelectorAll('[data-design-only]');
  const briefPlaceholders = {
    design: 'A landing page for a weekend cycling tour in rural Vermont.',
    conversation: "How do I make cold brew coffee at home?",
  };
  function applyKind() {
    const kind = kindSel ? kindSel.value : 'design';
    if (briefTA) briefTA.placeholder = briefPlaceholders[kind] || '';
    if (briefLabel) {
      briefLabel.firstChild.nodeValue = kind === 'conversation' ? 'Goal' : 'Brief';
    }
    conversationOnly.forEach((el) => {
      if (kind === 'conversation') el.removeAttribute('hidden');
      else el.setAttribute('hidden', '');
    });
    designOnly.forEach((el) => {
      if (kind === 'design') el.removeAttribute('hidden');
      else el.setAttribute('hidden', '');
    });
  }
  if (kindSel) {
    kindSel.addEventListener('change', applyKind);
    applyKind();
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    const kind = fd.get('kind') || 'design';
    const body = {
      brief: fd.get('brief'),
      max_iters: Number(fd.get('max_iters')),
      patience: Number(fd.get('patience')),
      tolerance: Number(fd.get('tolerance')),
      model: fd.get('model') || null,
      kind,
    };
    if (kind === 'conversation') {
      body.max_conversation_turns = Number(fd.get('max_conversation_turns')) || 5;
    } else {
      body.design_domain = fd.get('design_domain') || 'landing-page';
      body.evaluation_trials = Number(fd.get('evaluation_trials')) || 6;
      body.promotion_alpha = Number(fd.get('promotion_alpha')) || 0.05;
    }
    const token = fd.get('token');
    if (token) {
      body.token = token;
      localStorage.setItem(TOKEN_KEY, token);
    }
    const btn = form.querySelector('button');
    btn.disabled = true;
    status.textContent = 'starting…';
    try {
      const res = await fetch('/api/runs', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        if (res.status === 401) {
          localStorage.removeItem(TOKEN_KEY);
          throw new Error('invalid or missing access token');
        }
        throw new Error(await res.text());
      }
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
      const composite = Number(card.dataset.score);
      const sus = Number(card.dataset.sus);
      const eligible = card.dataset.eligible !== '0';
      return { iter: num, composite, sus, eligible };
    });
  }

  // Solid = primary metric (task completion for v2 design); dashed = SUS/CUS diagnostic.
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
    let cls = score >= 80 ? 'score-good' : score >= 60 ? 'score-ok' : 'score-bad';
    if (it.primary_metric === 'task_completion_rate' && !it.promotion_eligible) {
      cls += ' score-blocked';
    }
    const suggestions = (it.suggestions || []).map((s) =>
      `<li>${escapeHtml(s)}</li>`).join('');
    const kind = document.body.dataset.kind || 'design';
    let thumb, stats;
    if (kind === 'conversation') {
      thumb = `<a href="/runs/${runId}/iters/${it.iter}/transcript-view"
                  target="_blank" class="thumb thumb-transcript">
                 <div class="thumb-empty muted">open transcript →</div>
               </a>`;
      stats = `<span>CUS <b>${it.sus_score.toFixed(0)}</b></span>
               <span>penalty <b>${it.axe_penalty.toFixed(0)}</b></span>`;
    } else {
      thumb = `<a href="/runs/${runId}/iters/${it.iter}/site" target="_blank" class="thumb">
                 <img src="/runs/${runId}/iters/${it.iter}/screenshot" alt="Iter ${it.iter}" />
               </a>`;
      if (it.primary_metric === 'task_completion_rate') {
        stats = `<span>tasks <b>${(it.primary_score || 0).toFixed(0)}</b></span>
                 <span>SUS diagnostic <b>${it.sus_score.toFixed(0)}</b></span>
                 <span>guardrails <b>${it.promotion_eligible ? 'eligible' : 'blocked'}</b></span>
                 <span>decision <b>${it.promoted ? 'promoted' : 'rejected'}</b></span>`;
      } else {
        stats = `<span>SUS <b>${it.sus_score.toFixed(0)}</b></span>
                 <span>a11y penalty <b>${it.axe_penalty.toFixed(0)}</b></span>`;
      }
    }
    return `<article class="iter-card appearing" data-iter="${it.iter}"
      data-score="${score}" data-sus="${it.sus_score}"
      data-eligible="${it.promoted ? 1 : 0}">
      <header>
        <span class="iter-num">#${it.iter}</span>
        <span class="badge ${cls}">${score.toFixed(0)}</span>
      </header>
      ${thumb}
      <div class="stats">${stats}</div>
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
    const promotable = iters.filter((it) => it.eligible);
    if (!promotable.length) {
      statBest.textContent = '—';
      statBest.className = 'score-none';
      statBestIter.textContent = '—';
      statCount.textContent = iters.length;
      return;
    }
    let bestIter = promotable[0], best = promotable[0].composite;
    for (const it of promotable) {
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

  // Live progress indicator helpers.
  const progressEl = document.getElementById('progress-indicator');
  const progressText = document.getElementById('progress-text');
  function setProgress(iter, phase) {
    if (!progressEl) return;
    if (iter && phase) {
      progressEl.style.display = 'inline-flex';
      progressText.textContent = `iter ${iter} · ${phase}`;
    } else {
      progressEl.style.display = 'none';
      progressText.textContent = '';
    }
  }

  // Live updates via SSE — tell the server where we already are.
  const since = iters.length ? iters[iters.length - 1].iter : 0;
  const es = new EventSource(`/runs/${runId}/stream?since=${since}`);
  es.addEventListener('iteration', (e) => {
    const payload = JSON.parse(e.data);
    const it = payload.iter;
    // Append card
    grid.insertAdjacentHTML('beforeend', iterCardHtml(it));
    // Update chart data
    iters.push({
      iter: it.iter,
      composite: it.composite_score,
      sus: it.sus_score,
      eligible: it.promoted,
    });
    renderChart(iters);
    updateSummary(iters);
  });
  es.addEventListener('phase', (e) => {
    const { iter, phase } = JSON.parse(e.data);
    setProgress(iter, phase);
  });
  es.addEventListener('done', (e) => {
    const { run } = JSON.parse(e.data);
    const badge = document.querySelector('h1 .status');
    if (badge) {
      badge.className = `status status-${run.status}`;
      badge.textContent = run.status;
    }
    setProgress(null, null);
    es.close();
  });
  es.addEventListener('error', () => {
    // Connection drop; let the browser auto-reconnect unless the run is done.
  });
})();
