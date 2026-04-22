# design-gan

Autoresearch-style dual-agent loop that evolves single-page website designs. A
**generator** agent produces a site from a short brief; a **critic** agent
scores it on the System Usability Scale (SUS) alongside objective
accessibility signals (axe-core); the orchestrator feeds feedback back into
the generator and repeats until the composite score plateaus.

## Architecture

```
brief ──► generator ──► HTML ──► renderer ──► screenshot + DOM + axe
                                                     │
                                                     ▼
           best ◄─ scorer ◄─ SUS answers + suggestions ◄── critic
            │
            ▼
       sqlite + runs/
```

- **`generator.py`** — Claude writes a standalone HTML/CSS/JS document.
- **`renderer.py`** — Playwright headless Chromium: screenshot, DOM, axe-core.
- **`critic.py`** — Claude scores the screenshot on the 10-item SUS (Likert 1-5)
  and returns prioritized suggestions. Uses `messages.parse()` with a Pydantic
  schema so the response is always valid.
- **`scorer.py`** — Standard SUS scoring (0-100), blended with weighted axe
  violations into a composite score.
- **`orchestrator.py`** — The loop. Stops after `patience` iterations without
  a `tolerance`-point gain, or at `max_iters`.
- **`storage.py`** — SQLite run/iteration history.
- **`viewer.py`** — Minimal FastAPI viewer to browse iterations.

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
```

The viewer renders a dashboard with a run-start form, a live score chart, and
per-iteration cards (screenshot, SUS breakdown, feedback, suggestions). If
you start a run from the browser it streams new iterations in via SSE as
they complete — you can literally watch the site evolve.

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
single file, no JS framework, all screenshots inlined as base64. It's the fastest
way to show someone what design-gan does without giving them an API key.

```bash
pip install -e .
python scripts/build_site.py   # regenerates docs/index.html from the seed run
```

To publish: on GitHub, go to **Settings → Pages**, choose **Deploy from a branch**,
branch `main`, folder `/docs`. The site will be live at
`https://<you>.github.io/design-gan/`. Or point Vercel/Netlify at the `docs/`
directory.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

~90 tests covering the scorer, storage (schema + migration), the extractor
helpers, the orchestrator loop (with generator/critic/renderer faked), the
viewer's HTTP endpoints, and the CLI.

## Design notes

- **Critic sees the rendered page, not just code.** Code-only critique is
  cheap but correlates poorly with real usability.
- **Subjective + objective.** SUS alone is gameable; axe-core anchors the
  score to measurable a11y signals.
- **Convergence.** "No further improvements" is operationalized as
  `patience` iterations without a composite gain of at least `tolerance`
  points.
- **Caching.** The generator and critic both cache their system prompts with
  `cache_control: ephemeral`, so repeated iterations pay near-zero for the
  static instructions.
