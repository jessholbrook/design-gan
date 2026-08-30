# Design-GAN v2 handoff

Last updated: 2026-08-30

## Resume point

- Repository: <https://github.com/jessholbrook/design-gan>
- Branch: `main`
- Verified implementation commit: `48b317dde3f708591b3dd025f2d2da1dde5f9e65`
- Working tree at handoff: clean and synchronized with `origin/main`
- GitHub CI and Pages deployment for the handoff commit: passing
- Public showcase: <https://jessholbrook.github.io/design-gan/>
- Live application: <https://design-gan.fly.dev/>
- Fly app: `design-gan`, machine version 8, region `iad`, persistent volume mounted at `/data`

The v2 roadmap in `docs/roadmap.md` is implemented and merged. Design runs now
use repeated completion of frozen browser tasks as the primary product-quality
signal. SUS remains diagnostic feedback. Accessibility, browser correctness,
artifact boundaries, and evaluator execution are hard promotion guardrails.
The existing viewer, iteration cards, run history, and before/after scrubber are
preserved.

The static GitHub Pages showcase keeps the original visual style, v1 design
scrubber, and conversation example. It now also explains the v2 system using a
recorded behavioral-evaluation run and links to the live application.

## Start on another machine

Python 3.11 or newer is required.

```bash
git clone https://github.com/jessholbrook/design-gan.git
cd design-gan
git switch main
git pull --ff-only

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m playwright install chromium
cp .env.example .env
```

Add `ANTHROPIC_API_KEY` to `.env`, then verify the installation and launch the
local viewer:

```bash
python -c "from design_gan.cli import app; print('design-gan import OK')"
python -m pytest -q
python -m ruff check src tests
design-gan viewer
```

Open <http://127.0.0.1:8000/>. The evaluator-labeling workflow is at
<http://127.0.0.1:8000/evaluator-review>.

When opening the repository in Codex on the new machine, a sufficient resume
prompt is:

> Read `HANDOFF.md`, `README.md`, and `docs/roadmap.md`; verify the checkout and
> tests; then continue with the evaluator evidence-collection work described in
> the handoff. Preserve the frozen scenarios, primary metric, promotion
> guardrails, viewer, and scrubber.

If work should target the deployed environment, install and authenticate
`flyctl`, then use:

```bash
fly status --app design-gan
fly logs --app design-gan
fly deploy --app design-gan --remote-only
```

Do not recreate or rename the Fly volume. The deployed SQLite database and run
artifacts persist in the existing `data` volume at `/data`.

## Verified state

- `python -m pytest -q`: 294 passed
- `python -m ruff check src tests`: passed
- GitHub Actions CI: passed at the handoff commit
- GitHub Pages: published at the handoff commit
- Fly machine: running and healthy
- Public page: visually checked with no broken images or console errors

The current Fly deployment was built after the runtime/UI commit `a4b76a7`.
The commits between it and this handoff commit only refresh the static showcase,
stabilize Ruff's configured rule selection, and normalize styled CLI help tests;
they do not change deployed application behavior.

## Remaining work

The product roadmap is complete enough for real test runs. The remaining work is
evaluator evidence collection, not another framework milestone:

1. Run representative landing-page, lead-generation, and storefront loops.
2. Review balanced stored outcomes at `/evaluator-review`, using stable reviewer
   IDs and concrete rationales.
3. Accumulate the policy-v1 minimum of 24 qualifying real-run cases: at least
   eight per domain, at least three pass and three fail labels per domain, at
   least three source runs per domain, and no more than four qualifying cases
   from any one run.
4. Run `design-gan audit-evaluator-corpus`; keep experimental actor comparison
   fail-closed until this audit passes.
5. Only then benchmark a model-driven actor against the semantic baseline under
   explicit validity, cost, latency, and repeatability budgets.

The frozen scenario suites and promotion contract should not be adjusted in
response to individual candidates. Any evaluator change should be versioned and
validated against the labeled corpus first.

## State that does not travel through Git

- Local `runs/` data and local `.env` secrets are intentionally untracked.
- The deployed run history does travel operationally because it lives on the
  Fly volume, not because it is in the repository.
- On a fresh machine, use the live application to inspect existing deployed
  runs, or start new local runs after configuring `.env`.
