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

## Calibration and cross-run follow-on — complete

1. **Expanded evaluator validity corpus**
   - Corpus v3 adds generation-shaped failure classes: cookie controls mistaken
     for primary actions, action runtime errors, form spinners without success,
     explicit form receipt states, and pre-existing cart counts.
   - Semantic-v4 fixes those false positives and records 19/19 labeled cases in
     `docs/evaluator-benchmark-semantic-v4.json`.
   - These cases are regression seeds, not a claim that the corpus represents
     the frequency of failures in production-generated artifacts.
2. **Empirical calibration**
   - Three isolated replays per case produced 57/57 correct outcomes and 0%
     observed flakes. The machine-readable result is
     `docs/evaluator-calibration-semantic-v4.json`.
   - At α=0.05, an all-win exact sign test first clears the threshold with five
     discordant pairs (p=0.03125). The default is now five trials, not six.
   - `design-gan calibrate-evaluator` recomputes mismatch/flake rates and raises
     the recommended odd trial count when majority-error risk exceeds alpha.
3. **Cross-run incumbent ledger**
   - Every design run receives an explicit product optimization key or a stable
     normalized-brief key. Incumbents are separated by that key plus frozen
     domain, evaluator, and artifact-policy versions.
   - Search remains independent. Only after search ends are the final candidate
     and current incumbent freshly replayed on the same untouched holdout.
   - A fully passing challenger replaces the incumbent only when paired holdout
     outcomes clear minimum effect and significance. Ties, failures, and
     inconclusive audits retain the existing incumbent.
   - SQLite preserves incumbent lineage and challenge evidence. The CLI, API,
     run viewer, and scrubber expose the result without removing history.

## Provenance, saturation, and concurrency follow-on — complete

1. **Generated-run case capture**
   - `capture-evaluator-case` extracts a frozen task and exact HTML from a stored
     run iteration, records its run/iteration/task provenance and artifact hash,
     and requires an explicit operator pass/fail label.
   - Benchmark and calibration commands accept a case directory and reject
     invalid, duplicate, or artifact-policy-violating fixtures.
   - `/api/evaluator-cases` exposes case metadata and provenance without leaking
     the captured HTML. Fixtures remain local by default because they contain
     the complete generated artifact.
2. **Expanded sequestered holdouts**
   - Every concrete domain now freezes two development and two holdout scenarios.
   - The second holdout adds a mobile-keyboard condition, reducing the chance
     that a single pointer condition saturates while keeping the suite concrete.
3. **Concurrent challenge arbitration**
   - Ledger writes compare the incumbent evaluated by the challenger with the
     active incumbent inside one immediate SQLite transaction.
   - On conflict, the orchestrator fetches and replays the new incumbent once.
     A second conflict records a non-mutating inconclusive challenge, preserving
     the latest incumbent and bounded evaluator cost.
   - Conflict/retry evidence is persisted and visible through the run API,
     detail page, and existing scrubber.
4. **Recorded corpus-v4 calibration**
   - Corpus v4 adds mobile-keyboard success cases for all three domains.
   - Semantic-v4 records 22/22 correct outcomes in
     `docs/evaluator-benchmark-semantic-v4-corpus-v4.json`.
   - Three isolated replays record 66/66 correct outcomes, 0% observed flakes,
     and a five-trial recommendation in
     `docs/evaluator-calibration-semantic-v4-corpus-v4.json`.

## Uncertainty and composition follow-on — complete

1. **Descriptive confidence intervals**
   - Benchmark and calibration reports now record configurable two-sided Wilson
     intervals, defaulting to 95% confidence.
   - The 22/22 benchmark has a descriptive 85.1–100% interval. The 66/66 replay
     accuracy has a 94.5–100% interval, while 0/22 unstable cases still permits
     a 0–14.9% interval.
   - Reports explicitly state that the curated corpus is not a random sample of
     production artifacts. Repeated outcomes also share case structure, so the
     intervals must not be presented as production prevalence bounds.
2. **Corpus composition**
   - Machine-readable reports enumerate domain, expected outcome, behavior,
     interaction mode, exact viewport, and provenance-source counts.
   - CLI output surfaces the core domain, label, and provenance composition so
     a perfect headline score cannot hide an all-built-in or imbalanced corpus.
3. **Report contract**
   - Benchmark and calibration JSON now use report schema v2 while retaining
     evaluator corpus v4 and semantic actor v4.
   - `--confidence` changes the descriptive interval level without changing the
     frozen evaluator, corpus labels, promotion alpha, or trial recommendation.

## Operator review workflow — complete

1. **Run-history review queue**
   - `/evaluator-review` groups repeated trial outcomes by run, iteration, and
     frozen task, prioritizing evaluator failures while allowing successful
     outcomes to be included for balanced review.
   - Each item links to the existing sandboxed artifact, shows observed trials
     and runtime errors, and marks outcomes already captured in the local
     corpus. The run viewer and scrubber remain unchanged.
2. **Authenticated provenance capture**
   - `POST /api/evaluator-cases` reuses the optional viewer write token and
     requires an explicit operator pass/fail label plus a stable case id.
   - Captured fixtures contain the exact stored artifact, frozen task, artifact
     hash, and run provenance. The read APIs and page never expose artifact HTML.
   - Exclusive fixture creation returns a conflict rather than silently
     replacing an existing label.

## Provenance corpus admission — complete

1. **Auditable labels**
   - New CLI and viewer captures require a stable reviewer id and a 10–2000
     character rationale in addition to the pass/fail judgment.
   - Review metadata round-trips in the local fixture but remains absent from
     the open case-list API. Existing fixtures still load but do not qualify for
     actor admission when the audit trail is missing.
2. **Fail-closed readiness audit**
   - Policy v1 requires 24 qualifying real-run cases, eight per existing domain,
     three labels of each outcome per domain, and three distinct source runs per
     domain. No run may contribute more than four qualifying cases.
   - The audit excludes built-in cases, missing or mismatched provenance,
     mismatched artifact hashes, duplicate run/iteration/task provenance, and
     duplicate artifact/task evidence.
   - `audit-evaluator-corpus`, `/api/evaluator-corpus-readiness`, and the review
     page expose the same machine-readable blockers. A passing audit permits an
     actor comparison but is not a production-representativeness claim.

## Next research decisions

- Use the review queue during actual design runs to accumulate pass and failure
  examples; the repository intentionally does not fabricate or check in a
  private run fixture merely to claim real-run coverage.
- Reconsider a model actor only if it beats semantic-v4 on the expanded,
  provenance-backed corpus within explicit cost, latency, and repeatability
  budgets.
- Decide whether final actor adoption requires a second blinded reviewer and an
  adjudication record; policy v1 records one accountable operator judgment.
- Add another product domain only when it has a specific mutable artifact,
  frozen task contract, labeled failure cases, and a defensible north star.
