// Iteration scrubber. The shell is server-rendered by viewer.py's /scrub
// route; this script hydrates it from /api/runs/{id} and drives a slider that
// steps through iterations with the critique alongside each screenshot.
(() => {
  const root = document.querySelector('main.scrub');
  if (!root) return;
  const runId = document.body.dataset.scrubRunId;
  const kind = document.body.dataset.kind || 'design';
  if (!runId) return;

  const stageEl = document.getElementById('scrub-stage');
  const panelEl = document.getElementById('scrub-panel');
  const timelineEl = document.getElementById('scrub-timeline');

  let iters = [];
  let idx = 0;
  let bestIdx = 0;
  let mode = 'single'; // 'single' | 'prev' | 'best' (design runs only)
  let split = 0.5; // compare divider fraction, persists across iterations
  const transcriptCache = new Map();

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  }

  function scoreClass(v) {
    if (v == null) return 'score-none';
    return v >= 80 ? 'score-good' : v >= 60 ? 'score-ok' : 'score-bad';
  }

  function screenshotUrl(it) {
    return `/runs/${runId}/iters/${it.iter}/screenshot`;
  }

  // ----- Stage -----

  function compareTarget() {
    // The iteration the current one is being measured against, or null when
    // there's nothing meaningful to compare (first iter / already the peak).
    if (mode === 'prev') return idx > 0 ? iters[idx - 1] : null;
    if (mode === 'best') return idx !== bestIdx ? iters[bestIdx] : null;
    return null;
  }

  function renderSingleShot() {
    const it = iters[idx];
    stageEl.innerHTML = `
      <a class="scrub-shot" href="/runs/${runId}/iters/${it.iter}/site"
         target="_blank" rel="noopener"
         title="Open this iteration's generated HTML in a new tab">
        <img src="${screenshotUrl(it)}" alt="Iteration ${it.iter}" />
      </a>`;
  }

  function renderCompare(target) {
    const cur = iters[idx];
    stageEl.innerHTML = `
      <div class="scrub-compare" style="--f:${split}">
        <img class="scrub-compare-base" src="${screenshotUrl(target)}"
             alt="Iteration ${target.iter}" />
        <img class="scrub-compare-top" src="${screenshotUrl(cur)}"
             alt="Iteration ${cur.iter}" />
        <span class="scrub-compare-label left">#${cur.iter} now</span>
        <span class="scrub-compare-label right">#${target.iter} ${
      mode === 'best' ? 'peak' : 'prev'
    }</span>
        <div class="scrub-compare-handle" role="separator"
             aria-label="Drag to compare" tabindex="0"><span></span></div>
      </div>`;
    wireCompareDrag();
  }

  function wireCompareDrag() {
    const wrap = stageEl.querySelector('.scrub-compare');
    const handle = wrap.querySelector('.scrub-compare-handle');
    if (!wrap || !handle) return;

    const apply = (clientX) => {
      const r = wrap.getBoundingClientRect();
      split = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
      wrap.style.setProperty('--f', split);
    };
    const onMove = (e) => apply(e.clientX);
    const stop = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', stop);
    };
    handle.addEventListener('pointerdown', (e) => {
      e.preventDefault(); // don't start a scroll/selection
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', stop);
    });
    handle.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft') { split = Math.max(0, split - 0.05); }
      else if (e.key === 'ArrowRight') { split = Math.min(1, split + 0.05); }
      else return;
      e.preventDefault();
      e.stopPropagation();
      wrap.style.setProperty('--f', split);
    });
  }

  function renderStage() {
    const it = iters[idx];
    if (kind === 'conversation') {
      renderTranscriptStage(it);
      return;
    }
    const target = compareTarget();
    if (mode !== 'single' && target) renderCompare(target);
    else renderSingleShot();
  }

  function setMode(next) {
    mode = next;
    document.querySelectorAll('#scrub-modes button').forEach((b) => {
      b.classList.toggle('active', b.dataset.mode === mode);
    });
    renderStage();
  }

  function renderTranscriptStage(it) {
    const cached = transcriptCache.get(it.iter);
    if (cached) {
      stageEl.innerHTML = `<div class="scrub-transcript">${cached}</div>`;
      return;
    }
    stageEl.innerHTML = '<div class="scrub-loading muted">Loading transcript…</div>';
    fetch(`/runs/${runId}/iters/${it.iter}/transcript`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data) => {
        const turns = data.transcript || [];
        const html = turns
          .map(
            (t) =>
              `<div class="bubble bubble-${escapeHtml(t.role || '?')}">
                 <div class="bubble-role">${escapeHtml(t.role || '?')}</div>
                 <div class="bubble-text">${escapeHtml(t.content || '')}</div>
               </div>`
          )
          .join('');
        transcriptCache.set(it.iter, html);
        if (iters[idx].iter === it.iter) {
          stageEl.innerHTML = `<div class="scrub-transcript">${html}</div>`;
        }
      })
      .catch(() => {
        stageEl.innerHTML =
          '<div class="scrub-loading muted">No transcript for this iteration.</div>';
      });
  }

  function preloadNeighbours() {
    if (kind !== 'design') return;
    [idx - 1, idx + 1, bestIdx].forEach((j) => {
      if (j >= 0 && j < iters.length) {
        const img = new Image();
        img.src = screenshotUrl(iters[j]);
      }
    });
  }

  // ----- Panel -----

  function susRow(answers) {
    if (!Array.isArray(answers) || !answers.length) return '';
    const cells = answers
      .map(
        (v, i) =>
          `<li title="Q${i + 1}: ${v}/5">
             <span class="scrub-sus-bar">
               <span class="scrub-sus-fill" style="height:${(v / 5) * 100}%"></span>
             </span>
             <span class="scrub-sus-num">${v}</span>
           </li>`
      )
      .join('');
    return `<h3>SUS responses</h3><ul class="scrub-sus">${cells}</ul>`;
  }

  function deltaBadge() {
    if (idx === 0) return '<span class="scrub-delta muted">first iteration</span>';
    const d = iters[idx].composite_score - iters[idx - 1].composite_score;
    const sign = d > 0 ? '+' : '';
    const cls = d > 0 ? 'up' : d < 0 ? 'down' : 'flat';
    return `<span class="scrub-delta ${cls}">${sign}${d.toFixed(1)} vs #${
      iters[idx - 1].iter
    }</span>`;
  }

  function critiqueLabel() {
    return kind === 'conversation' ? 'CUS' : 'SUS';
  }

  function penaltyLabel() {
    return kind === 'conversation' ? 'penalty' : 'a11y penalty';
  }

  function criticBreakdown(it) {
    if (!Array.isArray(it.critic_breakdown) || !it.critic_breakdown.length) return '';
    const items = it.critic_breakdown
      .map((c) => {
        const sugg = (c.suggestions || [])
          .map((s) => `<li>${escapeHtml(s)}</li>`)
          .join('');
        return `<details class="scrub-critic">
            <summary>${escapeHtml(c.name || 'critic')}</summary>
            <p>${escapeHtml(c.feedback || '')}</p>
            ${sugg ? `<ul>${sugg}</ul>` : ''}
          </details>`;
      })
      .join('');
    return `<h3>Per-critic breakdown</h3>${items}`;
  }

  function askedToFix() {
    // The previous iteration's suggestions are exactly what the generator was
    // told to fix to produce *this* one — surfacing them makes the
    // feedback → change story legible.
    if (idx === 0) return '';
    const prev = iters[idx - 1];
    const sugg = (prev.suggestions || [])
      .map((s) => `<li>${escapeHtml(s)}</li>`)
      .join('');
    if (!sugg) return '';
    return `<details class="scrub-asked">
        <summary>What #${prev.iter}'s critic asked to fix →
          produced this iteration</summary>
        <ul>${sugg}</ul>
      </details>`;
  }

  function renderPanel() {
    const it = iters[idx];
    const suggestions = (it.suggestions || [])
      .map((s) => `<li>${escapeHtml(s)}</li>`)
      .join('');
    panelEl.innerHTML = `
      <div class="scrub-panel-head">
        <div>
          <span class="scrub-iter-label">Iteration</span>
          <span class="scrub-iter-num">#${it.iter}</span>
        </div>
        <span class="badge ${scoreClass(it.composite_score)}">
          ${it.composite_score.toFixed(0)}
        </span>
      </div>
      <div class="scrub-delta-row">${deltaBadge()}</div>
      <div class="scrub-stats">
        <div><span class="muted">${critiqueLabel()}</span>
          <b>${it.sus_score.toFixed(0)}</b></div>
        <div><span class="muted">${penaltyLabel()}</span>
          <b>${it.axe_penalty.toFixed(0)}</b></div>
        ${
          bestIdx === idx
            ? '<div><span class="muted">peak</span><b class="score-good">★</b></div>'
            : ''
        }
      </div>
      ${askedToFix()}
      <h3>Critic feedback</h3>
      <p class="scrub-feedback">${escapeHtml(it.feedback)}</p>
      ${
        suggestions
          ? `<h3>Suggestions for the next iteration</h3>
             <ul class="scrub-suggestions">${suggestions}</ul>`
          : ''
      }
      ${susRow(it.sus_answers)}
      ${criticBreakdown(it)}
    `;
  }

  // ----- Timeline -----

  function sparkline() {
    if (iters.length < 2) return '';
    const W = 100, H = 26;
    const scores = iters.map((it) => it.composite_score);
    const lo = Math.min(...scores, 0);
    const hi = Math.max(...scores, 1);
    const span = hi - lo || 1;
    const x = (i) => (i / (iters.length - 1)) * W;
    const y = (v) => H - ((v - lo) / span) * (H - 4) - 2;
    const pts = scores.map((v, i) => `${x(i)},${y(v)}`).join(' ');
    return `<svg class="scrub-spark" viewBox="0 0 ${W} ${H}"
              preserveAspectRatio="none" aria-hidden="true">
        <polyline points="${pts}" />
        <circle cx="${x(idx)}" cy="${y(scores[idx])}" r="2.5" class="cur" />
        <circle cx="${x(bestIdx)}" cy="${y(scores[bestIdx])}" r="2" class="best" />
      </svg>`;
  }

  function renderTimeline() {
    const last = iters.length - 1;
    timelineEl.innerHTML = `
      <button type="button" class="scrub-step" id="scrub-prev"
        aria-label="Previous iteration" ${idx === 0 ? 'disabled' : ''}>‹</button>
      <div class="scrub-track">
        ${sparkline()}
        <input type="range" id="scrub-range" min="0" max="${last}"
          step="1" value="${idx}" aria-label="Iteration" />
        <div class="scrub-ticks">
          ${iters
            .map(
              (it, j) =>
                `<button type="button" class="scrub-tick${
                  j === idx ? ' active' : ''
                }${j === bestIdx ? ' best' : ''}" data-j="${j}"
                  title="Iteration ${it.iter} · ${it.composite_score.toFixed(0)}">
                  #${it.iter}
                </button>`
            )
            .join('')}
        </div>
      </div>
      <button type="button" class="scrub-step" id="scrub-next"
        aria-label="Next iteration" ${idx === last ? 'disabled' : ''}>›</button>`;

    const range = document.getElementById('scrub-range');
    range.addEventListener('input', () => go(Number(range.value)));
    document.getElementById('scrub-prev').addEventListener('click', () => go(idx - 1));
    document.getElementById('scrub-next').addEventListener('click', () => go(idx + 1));
    timelineEl.querySelectorAll('.scrub-tick').forEach((b) => {
      b.addEventListener('click', () => go(Number(b.dataset.j)));
    });
  }

  // ----- Navigation -----

  function go(j) {
    const clamped = Math.max(0, Math.min(iters.length - 1, j));
    if (clamped === idx && stageEl.querySelector('img, .scrub-transcript')) {
      // Re-clicking the same tick: nothing to do.
      return;
    }
    idx = clamped;
    renderStage();
    renderPanel();
    renderTimeline();
    preloadNeighbours();
  }

  document.addEventListener('keydown', (e) => {
    if (e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
    if (e.key === 'ArrowLeft') { go(idx - 1); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { go(idx + 1); e.preventDefault(); }
    else if (e.key === 'Home') { go(0); e.preventDefault(); }
    else if (e.key === 'End') { go(iters.length - 1); e.preventDefault(); }
  });

  // ----- Boot -----

  fetch(`/api/runs/${runId}`)
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error('run not found'))))
    .then((data) => {
      iters = (data.iterations || []).slice().sort((a, b) => a.iter - b.iter);
      if (!iters.length) {
        stageEl.innerHTML =
          '<div class="scrub-loading muted">This run has no iterations yet.</div>';
        return;
      }
      bestIdx = iters.reduce(
        (best, it, j) =>
          it.composite_score > iters[best].composite_score ? j : best,
        0
      );
      idx = bestIdx;

      // Compare modes only make sense for design runs with >1 iteration.
      const modesEl = document.getElementById('scrub-modes');
      if (modesEl && kind === 'design' && iters.length > 1) {
        modesEl.removeAttribute('hidden');
        modesEl.querySelectorAll('button').forEach((b) => {
          b.addEventListener('click', () => setMode(b.dataset.mode));
        });
      }

      renderStage();
      renderPanel();
      renderTimeline();
      preloadNeighbours();
    })
    .catch((err) => {
      stageEl.innerHTML = `<div class="scrub-loading muted">Failed to load: ${escapeHtml(
        err.message
      )}</div>`;
    });
})();
