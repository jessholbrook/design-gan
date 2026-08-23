# design-gan

Autoresearch-style loop that evolves single-page website designs. A
**generator** produces a site from a short brief; Playwright then replays a
frozen behavioral task suite against it. Task completion is the primary
product-quality score. A **critic** still reports the System Usability Scale
(SUS) as diagnostic feedback, while axe-core accessibility and browser/runtime
correctness act as hard promotion guardrails.

![Scrubbing through a run — iteration #3 on the left, critic verdict on the right.](docs/images/scrubber-single.png)

## Architecture

```
brief ──► generator ──► HTML ──┬─► renderer ──► screenshot + DOM + axe
                               └─► browser evaluator ──► frozen task results
                                                        │
critic ──► SUS + feedback (diagnostic) ──────────────────┤
                                                        ▼
          best eligible ◄─ task score + hard guardrails ─ scorer
                 │
                 ▼
            sqlite + runs/ + viewer/scrubber
```

- **`generator.py`** — Claude writes a standalone HTML/CSS/JS document.
- **`renderer.py`** — Playwright headless Chromium: screenshot, DOM, axe-core
  (vendored into the package, so renders are deterministic and offline-safe).
- **`browser_evaluator.py`** — replays the run's frozen browser scenario from
  a clean page and records pass/fail evidence plus runtime errors. The first v2
  milestone deliberately contains one concrete scenario: find and activate the
  primary call to action and observe a meaningful response.
- **`critic.py`** — Claude scores the screenshot on the 10-item SUS (Likert 1-5)
  and returns prioritized suggestions. SUS is feedback, not the design
  north-star. The response contract is a fenced JSON block validated against a
  Pydantic schema, with one retry on malformed output.
- **`scorer.py`** — task completion rate (0-100) is the design primary metric.
  Candidates with critical/serious axe violations, an axe execution failure,
  JavaScript console errors, page errors, or evaluator action errors are marked
  ineligible for promotion rather than receiving a blended penalty.
- **`orchestrator.py`** — the loop. Only eligible candidates can become the
  best iteration; it stops after `patience` iterations without a
  `tolerance`-point primary-score gain, or at `max_iters`.
- **`storage.py`** — migration-safe SQLite run/iteration history, including the
  frozen suite, task evidence, diagnostic scores, and guardrail results.
- **`viewer.py`** — FastAPI viewer to browse iterations, plus a scrubber
  (`/runs/{id}/scrub`) for stepping through the evolution with a before/after
  compare slider.

## Setup

```bash
pip install -e .
playwright install chromium
cp .env.example .env  # add your ANTHROPIC_API_KEY
```

## Usage

```bash
# Launch the web UI: kick off runs, watch them live, browse history
design-gan viewer  # http://127.0.0.1:8000

# Or run one evolution loop from the terminal
design-gan run "A landing page for a weekend cycling tour in rural Vermont."
design-gan list-runs

# Write the best iteration's HTML (or system prompt, for conversation runs) to a file
design-gan export 3 --out best.html
```

The viewer renders a dashboard with a run-start form, a live score chart, and
per-iteration cards (screenshot, task score, promotion gates, diagnostic SUS,
feedback, and suggestions). If you start a run from the browser it streams new iterations in via SSE as
they complete — you can literally watch the site evolve.

![Run page — score over iterations and per-iteration cards.](docs/images/run-page.png)

Each run page has a **Scrub ▸** link to a dedicated scrubber: a timeline
slider with the screenshot on the left and the critic's verdict on the
right, updating as you drag (arrow keys work too). A **vs prev / vs best**
toggle overlays two iterations behind a draggable divider so you can see
exactly what changed, and each iteration surfaces the prior critic's
suggestions that produced it. Conversation runs scrub through transcripts
instead of screenshots.

![Scrubber compare mode — iteration #1 vs the peak iteration behind a draggable divider.](docs/images/scrubber-compare.png)

## Deploy to Fly.io

A `Dockerfile` and `fly.toml` are included. The Dockerfile bakes in Chromium
plus its Linux deps; runs persist to a mounted volume at `/data`.

One-time setup (from the repo root, with [flyctl](https://fly.io/docs/flyctl/)
installed and logged in):

```bash
# Claim an app name — edit fly.toml if the default is taken.
fly launch --no-deploy --copy-config

# Create the 1GB volume that backs SQLite + runs/ in the same region.
fly volumes create design_gan_data --size 1 --region iad

# Set your Anthropic key.
fly secrets set ANTHROPIC_API_KEY=sk-ant-...

# Deploy.
fly deploy
```

Once it's up:

```bash
# Seed the demo run so the dashboard isn't empty.
fly ssh console -C "design-gan demo"

# Tail logs while you try a real run from the web UI.
fly logs
```

If you hit OOM kills during renders, bump `[[vm]] memory = "2gb"` in `fly.toml`
and `fly deploy` again.

## Static showcase

A self-contained explainer page lives in [`docs/index.html`](docs/index.html) —
single file, no JS framework, all screenshots inlined as base64. Both runs on
the page are scrubbable (the same slider + compare interaction as the live
viewer, as inline progressive enhancement — it still reads fine with JS off).
Hand-edit the file directly; commit; GitHub Pages publishes in a minute at
`https://<you>.github.io/design-gan/`. On GitHub, enable it under
**Settings → Pages → Deploy from branch → `main` / `/docs`**.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

209 tests covering the browser-evaluator contract, primary scoring and
promotion gates, storage (schema + migration), the extractor
helpers, the orchestrator loop (with generator/critic/renderer faked), the
viewer's HTTP endpoints (including the scrubber route), and the CLI.

## Design notes

- **Critic sees the rendered page, not just code.** Code-only critique is
  cheap but correlates poorly with real usability.
- **One primary metric.** For design runs, browser-task completion is the only
  quantity used to rank eligible candidates. SUS and axe penalties remain
  visible diagnostics but are not averaged into the north-star.
- **Hard promotion gates.** Critical/serious accessibility failures and
  browser/runtime correctness errors block promotion even when task completion
  is high. Blocked candidates remain in history for diagnosis.
- **Frozen scenario.** The scenario definition is materialized once on the run
  record and replayed unchanged against every iteration. Each iteration writes
  `evaluation.json` alongside `site.html`, `screenshot.png`, `dom.html`, and
  `axe.json`.
- **Convergence.** "No further improvements" is operationalized as
  `patience` iterations without an eligible primary-score gain of at least
  `tolerance` points.
- **Greedy hill-climb.** Each iteration evolves from the best-scoring eligible
  iteration so far. When an iteration regresses, the next generation is
  re-seeded from the best artifact and its critique rather than drifting
  downhill from the regression. Until an iteration clears both guardrails, the
  loop keeps evolving the latest artifact so it can repair the observed failures.
- **Caching.** Generator and critic system prompts are static across
  iterations, so the Agent SDK's prompt caching keeps the repeated cost of
  those instructions near zero.

### Initial evaluator boundary

This milestone is intentionally not a generic action/assertion DSL. It covers
one ubiquitous landing-page behavior: the primary action must be discoverable,
enabled, and produce observable browser behavior (navigation, scrolling, a
dialog/new window, or changed visible content). The generator can explicitly
identify the main control with `data-primary-action`; otherwise the evaluator
uses semantic CTA/button/link signals.

Open design choices for subsequent milestones include how product-specific
task suites are authored and versioned, whether task attempts use a model-driven
browser actor or deterministic selectors, how many repeated trials are needed,
and what significance rule should gate promotion when completion becomes
stochastic. Those decisions are left open rather than hidden behind a premature
framework in this first slice.
