# Product optimization roadmap

The v2 north star is an autonomous loop that improves a bounded product artifact
against a frozen, product-specific scenario suite. One behavioral metric ranks
candidates; correctness and accessibility remain hard promotion gates.

## Implemented foundation

- A single-page HTML artifact evolves through a greedy candidate loop.
- Browser task completion is the design run's primary score.
- SUS remains diagnostic feedback.
- Accessibility and runtime correctness block promotion.
- Every candidate and its rendered artifacts remain available in SQLite and the
  run viewer/scrubber.

## Implemented milestone sequence

1. **Stable run contracts — complete**
   - Freeze a versioned evaluation plan on the run.
   - Freeze a versioned artifact policy on the run.
   - Record an explicit promotion decision for every candidate.
2. **Bounded artifacts — complete**
   - Enforce standalone, offline HTML and a maximum artifact size.
   - Treat boundary violations as promotion-blocking correctness failures.
3. **Repeated evaluation and significance — complete**
   - Replay every scenario for a fixed number of trials.
   - Promote improvements only when paired binary outcomes clear a one-sided
     exact sign test and the configured minimum effect.
4. **Candidate lineage — complete**
   - Persist the parent iteration used to generate each candidate.
   - Persist promotion status, reason, effect size, comparable trials, and
     p-value.
5. **Multiple product domains — complete**
   - Keep a landing-page primary-action profile.
   - Add a lead-generation form-completion profile.
   - Select a profile before the run; never change it mid-run.
6. **Operator controls and observability — complete**
   - Expose domain, trial count, and significance threshold in CLI/API/viewer.
   - Show frozen plans, artifact boundaries, lineage, and promotion evidence in
     the existing run viewer and scrubber.

## Deliberate boundaries

- Browser actions remain deterministic and semantic in this roadmap. A
  model-driven browser actor can be added as another evaluator version after a
  recorded benchmark demonstrates better validity.
- Significance uses paired task/trial outcomes, not noisy subjective scores.
- Conversation runs retain their current CUS path until they have their own
  domain-specific frozen scenario suite.
- The system does not mutate arbitrary repositories. The mutable artifact stays
  one standalone HTML document; widening that boundary requires a new artifact
  policy version.

## Completion criteria

- Existing databases migrate in place and legacy runs remain viewable.
- Existing viewer and scrubber routes remain intact.
- Every design run can explain: what was frozen, what was attempted, why a
  candidate was blocked or rejected, and why a candidate was promoted.
- The full automated suite and real Chromium smoke scenarios pass.

All criteria above are met by `codex/product-optimization-roadmap`.

## Next research decisions

- Benchmark semantic automation against a model-driven browser actor before
  introducing actor non-determinism into the north-star metric.
- Add controlled scenario variations only when they represent distinct user
  conditions rather than duplicate deterministic observations.
- Design holdout suites once each domain has enough scenarios to split without
  making either the development or holdout signal uninformative.
- Add another product domain only with a concrete bounded artifact and one
  defensible behavioral north-star; do not generalize the evaluator first.
