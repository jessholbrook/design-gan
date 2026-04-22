"""Build the static showcase site at docs/index.html.

Consumes the same pre-baked run as `design-gan demo` — no API key needed — and
emits a single self-contained HTML file. Screenshots are inlined as base64
data URIs so the resulting page is drop-in hostable (GitHub Pages, Vercel,
any static host).
"""

from __future__ import annotations

import base64
import html
import json
import tempfile
from pathlib import Path

from design_gan.demo import seed_demo
from design_gan.storage import Storage

REPO_URL = "https://github.com/jessholbrook/design-gan"

# Fallback for runs_dir-style on-disk artifacts.
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs"


def _escape(s: str) -> str:
    return html.escape(s)


def _score_class(score: float | None) -> str:
    if score is None:
        return "score-none"
    if score >= 80:
        return "score-good"
    if score >= 60:
        return "score-ok"
    return "score-bad"


def _b64_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


SUS_ITEMS_SHORT = [
    "Would use frequently",
    "Unnecessarily complex",
    "Easy to use",
    "Needs tech support",
    "Well integrated",
    "Too inconsistent",
    "Learn quickly",
    "Cumbersome",
    "Confident using",
    "Need to learn a lot",
]


def _sus_breakdown_html(answers: list[int]) -> str:
    """Render the 10 SUS Likert answers as a compact bar row."""
    cells = []
    for i, (label, val) in enumerate(zip(SUS_ITEMS_SHORT, answers)):
        # Positive items (odd 1-indexed -> even 0-indexed) contribute (val-1);
        # negative items contribute (5-val). Color by contribution on 0-4.
        positive = (i % 2 == 0)
        contrib = (val - 1) if positive else (5 - val)
        pct = int((contrib / 4) * 100)
        sign = "+" if positive else "−"
        cells.append(
            f'<li title="{_escape(label)}: {val}/5 ({sign})">'
            f'<span class="sus-bar"><span class="sus-fill" style="height:{pct}%"></span></span>'
            f'<span class="sus-num">{val}</span>'
            f'</li>'
        )
    return f'<ul class="sus-row">{"".join(cells)}</ul>'


def _score_chart(iters: list[dict]) -> str:
    """SVG line chart of composite + SUS over iterations."""
    if not iters:
        return ""
    W, H = 800, 240
    padL, padR, padT, padB = 40, 16, 16, 28
    max_iter = max(len(iters), 5)

    def x(i: int) -> float:
        return padL + ((i - 1) / max(1, max_iter - 1)) * (W - padL - padR)

    def y(v: float) -> float:
        return padT + (1 - v / 100) * (H - padT - padB)

    grid_lines = "\n".join(
        f'<line class="grid-line" x1="{padL}" y1="{y(v):.1f}" x2="{W - padR}" y2="{y(v):.1f}" />'
        f'<text class="axis-label" x="6" y="{y(v) + 4:.1f}">{v}</text>'
        for v in (0, 25, 50, 75, 100)
    )

    x_labels = "\n".join(
        f'<text class="axis-label" x="{x(it["iter"]):.1f}" y="{H - padB + 16}" '
        f'text-anchor="middle">#{it["iter"]}</text>'
        for it in iters
    )

    compo_pts = " ".join(f'{x(it["iter"]):.1f},{y(it["composite_score"]):.1f}' for it in iters)
    sus_pts = " ".join(f'{x(it["iter"]):.1f},{y(it["sus_score"]):.1f}' for it in iters)
    dots = "\n".join(
        f'<g><circle class="point" cx="{x(it["iter"]):.1f}" cy="{y(it["composite_score"]):.1f}" r="4" />'
        f'<text class="point-label" x="{x(it["iter"]):.1f}" y="{y(it["composite_score"]) - 10:.1f}" '
        f'text-anchor="middle">{it["composite_score"]:.0f}</text></g>'
        for it in iters
    )

    return f"""<svg viewBox="0 0 {W} {H}" class="chart" preserveAspectRatio="xMidYMid meet">
  {grid_lines}
  <polyline class="line-sus" points="{sus_pts}" />
  <polyline class="line-composite" points="{compo_pts}" />
  {dots}
  {x_labels}
  <g class="legend" transform="translate({W - 180}, {padT + 6})">
    <line class="line-composite" x1="0" y1="6" x2="24" y2="6" />
    <text x="30" y="10">composite</text>
    <line class="line-sus" x1="90" y1="6" x2="114" y2="6" />
    <text x="120" y="10">SUS</text>
  </g>
</svg>"""


def _iter_card(i: int, it: dict, screenshot_b64: str, site_html: str) -> str:
    sus_html = _sus_breakdown_html(it["sus_answers"])
    suggestions = "".join(f"<li>{_escape(s)}</li>" for s in it["suggestions"])
    site_src = base64.b64encode(site_html.encode("utf-8")).decode("ascii")
    score = it["composite_score"]
    cls = _score_class(score)
    # Color for the header score badge.
    return f"""<article class="iter">
  <header>
    <div class="iter-meta">
      <span class="iter-label">Iteration</span>
      <span class="iter-num">#{it['iter']}</span>
    </div>
    <div class="iter-score">
      <span class="badge {cls}">{score:.0f}</span>
      <div class="iter-score-breakdown">
        <div><span>SUS</span><b>{it['sus_score']:.0f}</b></div>
        <div><span>a11y penalty</span><b>{it['axe_penalty']:.0f}</b></div>
      </div>
    </div>
  </header>
  <div class="iter-body">
    <a class="iter-thumb" href="data:text/html;base64,{site_src}" target="_blank" rel="noopener"
       title="Open this iteration's generated HTML">
      <img src="data:image/png;base64,{screenshot_b64}" alt="Screenshot of iteration {it['iter']}" />
    </a>
    <div class="iter-detail">
      <h3>Critic feedback</h3>
      <p class="feedback">{_escape(it['feedback'])}</p>
      <h3>SUS answers <span class="muted">(1–5 Likert, 10 items)</span></h3>
      {sus_html}
      <h3>Suggestions sent to the next generator</h3>
      <ul class="suggestions">{suggestions}</ul>
    </div>
  </div>
</article>"""


def build(output_dir: Path = OUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        run_id = seed_demo(tmp_path)
        store = Storage(tmp_path / "design-gan.sqlite")
        run = store.get_run(run_id)
        iters = store.iterations_for_run(run_id)
        assert run is not None and iters, "demo seed returned no data"

        # Pull screenshots + site HTML off disk before the tmp dir is cleaned up.
        iter_assets: list[tuple[str, str]] = []
        for it in iters:
            art_dir = Path(it["artifacts_dir"])
            png_b64 = _b64_png(art_dir / "screenshot.png")
            site_html = (art_dir / "site.html").read_text(encoding="utf-8")
            iter_assets.append((png_b64, site_html))

    best = max(iters, key=lambda it: it["composite_score"])
    first = min(iters, key=lambda it: it["iter"])
    delta = best["composite_score"] - first["composite_score"]

    chart_svg = _score_chart(iters)
    iter_cards = "\n".join(
        _iter_card(i, it, png_b64, site_html)
        for i, (it, (png_b64, site_html)) in enumerate(zip(iters, iter_assets))
    )

    # Machine-readable dump for curious readers who dig in via DevTools.
    dataset = [
        {
            "iter": it["iter"],
            "sus_score": it["sus_score"],
            "axe_penalty": it["axe_penalty"],
            "composite_score": it["composite_score"],
            "sus_answers": it["sus_answers"],
            "feedback": it["feedback"],
            "suggestions": it["suggestions"],
        }
        for it in iters
    ]

    html_doc = PAGE_TEMPLATE.format(
        repo_url=REPO_URL,
        brief=_escape(run["brief"].replace("DEMO: ", "")),
        best_score=f"{best['composite_score']:.0f}",
        first_score=f"{first['composite_score']:.0f}",
        delta=f"{delta:+.0f}",
        best_iter=best["iter"],
        iter_count=len(iters),
        chart_svg=chart_svg,
        iter_cards=iter_cards,
        dataset_json=json.dumps(dataset, indent=2),
    )

    out = output_dir / "index.html"
    out.write_text(html_doc, encoding="utf-8")
    # Empty .nojekyll so GitHub Pages serves everything (including _-prefixed
    # paths if ever added) without Jekyll's special rules.
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")
    return out


PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>design-gan — a dual-agent loop that evolves website designs</title>
  <meta name="description" content="An autoresearch-style experiment: a Claude generator writes single-page sites, a Claude critic scores them on the System Usability Scale, and the loop feeds feedback back in until the score plateaus." />
  <style>
    :root {{
      --bg: #faf9f6;
      --panel: #ffffff;
      --border: #e7e5e4;
      --text: #18181b;
      --muted: #6b7280;
      --accent: #2563eb;
      --accent-soft: #eff6ff;
      --good: #059669;
      --ok: #d97706;
      --bad: #dc2626;
      --shadow: 0 1px 2px rgba(0, 0, 0, 0.04), 0 4px 12px rgba(0, 0, 0, 0.04);
    }}

    * {{ box-sizing: border-box; }}

    html, body {{ margin: 0; }}

    body {{
      font: 16px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      color: var(--text);
      background: var(--bg);
    }}

    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}

    .muted {{ color: var(--muted); }}

    .container {{ max-width: 960px; margin: 0 auto; padding: 48px 24px; }}

    /* Hero */
    .hero {{ padding-bottom: 32px; border-bottom: 1px solid var(--border); }}
    .kicker {{
      color: var(--muted);
      letter-spacing: 0.16em;
      text-transform: uppercase;
      font-size: 12px;
      font-weight: 600;
    }}
    h1 {{
      font-size: clamp(32px, 5vw, 48px);
      line-height: 1.1;
      margin: 12px 0 16px;
      letter-spacing: -0.02em;
    }}
    .lede {{
      font-size: 18px;
      color: var(--muted);
      max-width: 620px;
      margin: 0 0 24px;
    }}
    .actions {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    .btn {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 16px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 14px;
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
    }}
    .btn:hover {{ border-color: var(--accent); text-decoration: none; }}
    .btn.primary {{
      background: var(--text);
      color: var(--bg);
      border-color: var(--text);
    }}
    .btn.primary:hover {{ opacity: 0.9; }}

    /* Headline stats */
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin-top: 32px;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
    }}
    .stat-label {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .stat-value {{
      font-size: 28px;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
      letter-spacing: -0.01em;
      margin-top: 4px;
    }}
    .stat-value.good {{ color: var(--good); }}

    /* Sections */
    section {{ margin-top: 56px; }}
    section h2 {{
      font-size: 26px;
      letter-spacing: -0.01em;
      margin: 0 0 12px;
    }}
    section p {{ margin: 0 0 16px; }}
    section p.lead {{ color: var(--muted); font-size: 17px; }}

    /* Architecture diagram */
    .diagram {{
      display: block;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
      color: var(--text);
      white-space: pre;
      overflow-x: auto;
    }}

    /* Chart */
    .chart-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      margin-top: 16px;
      box-shadow: var(--shadow);
    }}
    .chart {{ width: 100%; height: 240px; display: block; }}
    .chart .grid-line {{ stroke: #e5e4e1; stroke-width: 1; }}
    .chart .axis-label {{ fill: var(--muted); font-size: 11px; font-family: inherit; }}
    .chart .line-composite {{ fill: none; stroke: var(--accent); stroke-width: 2.5; }}
    .chart .line-sus {{ fill: none; stroke: var(--muted); stroke-width: 1.5; stroke-dasharray: 4 3; }}
    .chart .point {{ fill: var(--accent); stroke: #fff; stroke-width: 2; }}
    .chart .point-label {{ fill: var(--text); font-size: 11px; font-weight: 600; font-family: inherit; }}
    .chart .legend text {{ font-size: 11px; fill: var(--muted); font-family: inherit; }}

    /* Iteration cards */
    .iter-stack {{ display: flex; flex-direction: column; gap: 20px; margin-top: 16px; }}
    .iter {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }}
    .iter > header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      background: #fafaf8;
      gap: 12px;
    }}
    .iter-label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; display: block; }}
    .iter-num {{ font-size: 20px; font-weight: 700; font-variant-numeric: tabular-nums; }}
    .iter-score {{ display: flex; align-items: center; gap: 16px; }}
    .iter-score-breakdown {{ display: flex; gap: 16px; font-size: 12px; color: var(--muted); }}
    .iter-score-breakdown span {{ display: block; text-transform: uppercase; letter-spacing: 0.04em; font-size: 10px; }}
    .iter-score-breakdown b {{ font-size: 16px; color: var(--text); font-variant-numeric: tabular-nums; }}
    .badge {{
      display: inline-block;
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 18px;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }}
    .badge.score-good {{ background: #d1fae5; color: #065f46; }}
    .badge.score-ok {{ background: #fef3c7; color: #92400e; }}
    .badge.score-bad {{ background: #fee2e2; color: #991b1b; }}
    .badge.score-none {{ background: #e5e7eb; color: #374151; }}

    .iter-body {{
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 20px;
      padding: 20px;
    }}
    .iter-thumb {{
      display: block;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      line-height: 0;
      aspect-ratio: 16 / 9;
    }}
    .iter-thumb img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      object-position: top left;
      transition: transform 0.3s ease;
    }}
    .iter-thumb:hover img {{ transform: scale(1.02); }}
    .iter-detail h3 {{ font-size: 13px; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); margin: 0 0 8px; font-weight: 600; }}
    .iter-detail h3:not(:first-child) {{ margin-top: 18px; }}
    .iter-detail .feedback {{ margin: 0; }}
    .suggestions {{ margin: 0; padding-left: 20px; }}
    .suggestions li {{ margin-bottom: 4px; }}

    /* SUS breakdown row */
    .sus-row {{
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      grid-template-columns: repeat(10, 1fr);
      gap: 6px;
    }}
    .sus-row li {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
    }}
    .sus-bar {{
      width: 100%;
      height: 44px;
      background: #f1efe9;
      border-radius: 4px;
      display: flex;
      align-items: flex-end;
      overflow: hidden;
    }}
    .sus-fill {{
      width: 100%;
      background: var(--accent);
      border-radius: 4px;
      min-height: 2px;
    }}
    .sus-num {{ font-size: 10px; color: var(--muted); font-variant-numeric: tabular-nums; }}

    /* Code blocks */
    pre, code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
    pre {{
      background: #0f172a;
      color: #e2e8f0;
      padding: 16px 18px;
      border-radius: 8px;
      overflow-x: auto;
      line-height: 1.5;
    }}
    pre code {{ background: transparent; padding: 0; color: inherit; }}
    p code {{
      background: #eef2ff;
      padding: 1px 6px;
      border-radius: 4px;
      font-size: 13px;
    }}

    /* Footer */
    footer {{ margin-top: 72px; padding: 24px 0; border-top: 1px solid var(--border); color: var(--muted); font-size: 14px; }}

    details.dataset {{
      margin-top: 16px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 16px;
    }}
    details.dataset summary {{ cursor: pointer; color: var(--muted); font-size: 13px; }}
    details.dataset pre {{ margin-top: 10px; max-height: 320px; }}

    @media (max-width: 720px) {{
      .iter-body {{ grid-template-columns: 1fr; }}
      .sus-row {{ grid-template-columns: repeat(5, 1fr); }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <section class="hero">
      <span class="kicker">An experiment · autoresearch × design</span>
      <h1>Two Claudes, one critic, and a website that learns.</h1>
      <p class="lede">
        <strong>design-gan</strong> is a dual-agent loop: a <em>generator</em>
        writes a single-page website from a brief, a <em>critic</em> scores it
        on the System Usability Scale, and the feedback rolls forward into the
        next draft — until the composite score plateaus.
      </p>
      <div class="actions">
        <a class="btn primary" href="{repo_url}">View on GitHub</a>
        <a class="btn" href="#the-run">Skip to the run &rarr;</a>
      </div>

      <div class="stats">
        <div class="stat">
          <div class="stat-label">Best composite</div>
          <div class="stat-value good">{best_score}</div>
        </div>
        <div class="stat">
          <div class="stat-label">Starting composite</div>
          <div class="stat-value">{first_score}</div>
        </div>
        <div class="stat">
          <div class="stat-label">Delta</div>
          <div class="stat-value good">{delta}</div>
        </div>
        <div class="stat">
          <div class="stat-label">Iterations</div>
          <div class="stat-value">{iter_count}</div>
        </div>
      </div>
    </section>

    <section id="the-idea">
      <h2>The idea</h2>
      <p class="lead">
        Good-looking demos don't prove usability. The only way to know whether a
        generated site <em>actually works</em> is to put it in front of a user —
        but that doesn't scale. <strong>design-gan</strong> approximates that
        loop with a second model.
      </p>
      <p>
        A <strong>generator</strong> agent writes a standalone HTML document
        from the brief. A headless Chromium takes a screenshot, dumps the DOM,
        and runs <a href="https://github.com/dequelabs/axe-core">axe-core</a>
        for a hard-grounded accessibility check. A <strong>critic</strong>
        agent then sees the <em>rendered page</em> — not the code — and fills
        out the 10-item <a href="https://measuringu.com/sus/">System Usability
        Scale</a>, plus writes prose feedback and a short list of concrete
        next-step suggestions.
      </p>
      <p>
        The suggestions feed the next generator call. The loop stops after a
        configurable patience window of iterations that produce no meaningful
        composite-score gain, or at a hard iteration cap.
      </p>
    </section>

    <section id="how-it-works">
      <h2>How a single iteration works</h2>
      <pre class="diagram">brief ──▶ generator ──▶ HTML ──▶ headless Chromium
                                          │
                                          ├─▶ screenshot
                                          ├─▶ DOM
                                          └─▶ axe-core violations
                                                    │
                                                    ▼
                         critic ◀── System Usability Scale
                           │
                           ├─▶ SUS answers (10 × 1–5)
                           ├─▶ prose feedback
                           └─▶ prioritized suggestions ──▶ next generator call</pre>
      <p>
        The composite score is the standard SUS score (0–100) minus a weighted
        axe-core penalty. Accessibility alone can't dominate — the penalty is
        capped — but it keeps the critic honest: "looks nice" can't hide a
        colour-contrast failure or missing alt text.
      </p>
    </section>

    <section id="the-run">
      <h2>A run, iteration by iteration</h2>
      <p class="lead">
        Brief: &ldquo;{brief}&rdquo;
      </p>
      <p>
        Four iterations, from a wall of unstyled text to something that would
        survive a real usability study. The critic's feedback and suggestion
        list are verbatim — these are the tokens that shaped the next draft.
        Click any screenshot to open that iteration's actual generated HTML in
        a new tab.
      </p>

      <div class="chart-card">
        {chart_svg}
      </div>

      <div class="iter-stack">
        {iter_cards}
      </div>
    </section>

    <section id="design-notes">
      <h2>Design notes</h2>
      <ul>
        <li><strong>The critic sees the rendered page, not the code.</strong> Code-only critique is cheap but correlates poorly with real usability.</li>
        <li><strong>Subjective + objective.</strong> SUS alone is gameable; axe-core anchors the score to measurable a11y signals.</li>
        <li><strong>Convergence, operationalised.</strong> &ldquo;No further improvements&rdquo; = <code>patience</code> iterations in a row without a composite gain of at least <code>tolerance</code> points.</li>
        <li><strong>Structured output.</strong> The critic uses Pydantic-validated JSON so a malformed response triggers a retry with a stricter nudge.</li>
        <li><strong>Prompt caching.</strong> System prompts are cached as <code>ephemeral</code>, so iteration <em>N+1</em> pays near-zero for the static instructions.</li>
      </ul>
    </section>

    <section id="run-it">
      <h2>Run it yourself</h2>
      <pre><code>pip install -e .
playwright install chromium
cp .env.example .env  # add your ANTHROPIC_API_KEY

# Web UI — kick off runs, watch them stream in live
design-gan viewer      # http://127.0.0.1:8000

# Or one evolution loop from the terminal
design-gan run "A landing page for a weekend cycling tour in rural Vermont."</code></pre>
      <p>
        Source, setup details, and a Fly.io deploy config are in the
        <a href="{repo_url}">GitHub repository</a>.
      </p>

      <details class="dataset">
        <summary>Raw dataset for this run (JSON)</summary>
        <pre><code>{dataset_json}</code></pre>
      </details>
    </section>

    <footer>
      <p>
        An autoresearch-style experiment. Generator and critic are both Claude
        Sonnet. The run shown is seeded — no API calls were made to render this
        page. <a href="{repo_url}">Source on GitHub</a>.
      </p>
    </footer>
  </div>
</body>
</html>
"""


if __name__ == "__main__":
    out = build()
    print(f"wrote {out}")
