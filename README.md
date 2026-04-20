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
