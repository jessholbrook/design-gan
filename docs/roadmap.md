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

## Evaluator-rigor follow-on — complete

1. **Recorded validity benchmark**
   - Added a labeled Chromium corpus covering expected passes and failures,
     distractor forms, mobile visibility, keyboard activation, and cart-state
     false positives.
   - The semantic-v3 actor scores 13/13 on corpus v2. The recorded report is
     `docs/evaluator-benchmark-semantic-v3.json`.
   - A model-driven actor remains outside the north-star path: it cannot show
     better validity on the current corpus, and would add cost and variance.
2. **Controlled scenario variations**
   - Domain profiles now freeze concrete pointer/keyboard and desktop/mobile
     conditions rather than treating identical reruns as distinct scenarios.
   - Trials still repeat each condition to detect runtime instability.
3. **Non-adaptive holdout audit**
   - Two development scenarios drive iteration feedback and promotion.
   - One holdout scenario is omitted from generator feedback and runs exactly
     once against the final promoted artifact.
   - Holdout score, pass/fail, evidence, timestamp, and audited iteration are
     persisted on the run and shown in the viewer and scrubber.
4. **Third concrete domain**
   - Added a single-product storefront profile whose north star is successful
     add-to-cart completion with visible cart evidence.
   - Generic visual change does not count as cart completion.

## Next research decisions

- Grow the labeled benchmark with failures observed in real generated runs;
  reconsider a model actor only if it beats the semantic baseline on a future
  corpus while meeting cost and repeatability budgets.
- Calibrate trial count and significance thresholds from empirical flake rates
  rather than treating six trials as permanently optimal.
- Add a cross-run incumbent/holdout ledger so a new run can challenge a
  previously verified artifact without reusing its development feedback.
