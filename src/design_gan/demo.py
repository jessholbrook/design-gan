"""Seed a demo run so the viewer has real content to show without an API key.

The artifacts bundled in ``demo_assets/`` were captured from an actual six-iteration
run of the loop against the brief shown below. We ship the first four iterations
(baseline through the peak-then-regression) because they tell the clearest story.
"""

from __future__ import annotations

from pathlib import Path

from . import storage

BRIEF = (
    "DEMO: A landing page for a weekend cycling tour in rural Vermont "
    "(captured from a real six-iteration run)."
)

_ASSETS = Path(__file__).parent / "demo_assets"


# Real SUS answers, a11y penalty, critic feedback, and suggestions from
# run_0001 iterations 1-4. Scores aren't stored here — they're computed from
# sus_answers + synthetic violations shaped to match the recorded axe_penalty
# (see _violations_matching_penalty below) so the saved composite matches what
# the real critic produced.
_ITERS = [
    {
        "sus_answers": [2, 1, 4, 1, 4, 2, 5, 1, 4, 1],
        "axe_penalty": 30.0,
        "feedback": (
            "Green Mountain Gravel is a well-crafted, visually cohesive landing page "
            "with a clear information hierarchy, prominent CTAs, and an attractive "
            "thematic design. The fixed nav, stat bar, and section structure make "
            "orientation effortless for any user. However, axe-core flags 8 nodes "
            "with insufficient color contrast (particularly the ~78% white subtitle "
            "text over dark green and small uppercase labels) and 3 in-body links "
            "indistinguishable without color, both of which meaningfully reduce "
            "usability for low-vision or color-blind users. The site's scope is "
            "inherently single-purpose, limiting 'frequent use' appeal, but within "
            "its task domain it performs confidently."
        ),
        "suggestions": [
            "Fix the 8 color-contrast failures — raise subtitle and body text on the dark "
            "green hero to at least 4.5:1; the rgba(255,255,255,.78) value (~3.8:1) is the "
            "likely culprit, bumping opacity to 0.92+ would resolve it.",
            "Make in-body links distinguishable without color (e.g., add an underline or "
            "bold weight) to address the 3 link-in-text-block violations, ensuring they're "
            "usable in grayscale or by color-blind visitors.",
            "Add a visible focus indicator everywhere — nav links use `outline: none` on "
            "hover/focus which removes the ring for keyboard users; replace with a "
            "consistent amber or white outline.",
            "The hero stat bar ('2 Days / 62 mi / 4,800' / 40 Riders Max / Sept') is "
            "visually strong but cut off at the very bottom fold on typical viewports — "
            "nudge it fully into view or add a subtle scroll-down hint so users discover it.",
            "ARIA role mismatches on 3 nodes and an `aside` inside a landmark should be "
            "cleaned up to avoid screen-reader confusion — audit the decorative SVG and "
            "sidebar elements to ensure roles match their semantic purpose.",
            "Consider adding a sticky 'Spots remaining' counter or a short social-proof "
            "line (e.g., 'Sold out in 2024') near the CTA — the page's visual quality "
            "earns trust but gives no urgency signal for a capacity-limited tour.",
        ],
    },
    {
        "sus_answers": [2, 2, 4, 1, 4, 2, 5, 1, 4, 1],
        "axe_penalty": 15.0,
        "feedback": (
            "This is a polished, single-purpose landing page with strong visual hierarchy, "
            "a coherent green-and-amber palette, and well-placed CTAs that communicate the "
            "event's value quickly. The stats bar at the hero bottom is a smart design "
            "choice that surfaces key decision-making facts without requiring a scroll. "
            "The main drags on the score are the 5 axe-core color-contrast failures (real "
            "barriers for low-vision users), the inherently low 'use frequently' ceiling "
            "of an event registration page, and the limited confidence a first-time "
            "visitor might have in the registration flow which isn't visible above the fold."
        ),
        "suggestions": [
            "Resolve all 5 color-contrast violations flagged by axe-core — audit amber "
            "text on dark-green backgrounds and light-gray body text on the cream "
            "background, as these are the most likely offenders.",
            "The 'SCROLL' cue is tiny and low-contrast against the hero background; "
            "increase its size or replace with a more prominent animated chevron so users "
            "know substantial content awaits below the fold.",
            "Add a brief difficulty or fitness-level indicator near the '62 mi per day / "
            "4,800 ft climb' stats — prospective riders need to self-qualify before "
            "registering, and omitting this creates uncertainty.",
            "The 'Register Now' nav CTA and 'Reserve Your Spot' hero CTA both lead to the "
            "same action but use different labels — unify the wording to reduce cognitive "
            "friction.",
            "Include a visible price anchor (even a 'From $X' teaser) near the primary "
            "CTA; without it, users may hesitate to click because they don't know what "
            "commitment they're walking into.",
            "The desktop nav has five links plus a CTA — ensure a hamburger or equivalent "
            "exists for mobile viewports, as the current layout will overflow on small "
            "screens.",
        ],
    },
    {
        "sus_answers": [2, 1, 5, 1, 4, 1, 5, 1, 4, 1],
        "axe_penalty": 12.0,
        "feedback": (
            "Green Mountain Gravel is a well-crafted, single-purpose landing page with "
            "strong visual hierarchy, a cohesive green-and-amber design system, and clear "
            "CTAs that guide visitors straight to registration. The fixed nav with "
            "descriptive section labels, an upfront price anchor, and an urgency signal "
            "all reduce friction effectively. Four serious color-contrast violations "
            "(flagged by axe-core) affect legibility for some users, and the hero headline "
            "'Two Hundred Miles' appears to conflict with the stats bar showing '62 mi,' "
            "which could momentarily undermine trust. Overall this is an above-average "
            "marketing page that would perform well in a usability study, but small "
            "inconsistencies and accessibility gaps keep it from a top score."
        ),
        "suggestions": [
            "Fix all 4 axe-core color-contrast violations — likely the small eyebrow/label "
            "text and the price-anchor line against the dark hero background; bump type "
            "size or darken/lighten the foreground color to achieve 4.5:1 minimum.",
            "Reconcile the headline claim 'Two Hundred Miles' with the stats bar value of "
            "'62 mi' — if 62 mi is per-day average, add a '/day' label; otherwise correct "
            "the headline to avoid eroding trust for detail-oriented cyclists.",
            "The stats bar is cut off at the bottom of the viewport in the initial view — "
            "either reduce hero padding so the stats row is fully visible above the fold, "
            "or surface the most critical stats (distance, elevation, dates) in the hero "
            "body copy itself.",
            "Add social proof (past-rider photos, 1–2 short testimonials, or a press "
            "mention badge) in the hero or immediately below the fold — for a ~$449 "
            "purchase decision, trust signals near the primary CTA materially increase "
            "conversion.",
            "The mobile hamburger menu is present in CSS but hidden on desktop; verify it "
            "is fully keyboard- and screen-reader-accessible on small viewports, as the "
            "nav links carry the site's entire wayfinding.",
            "Consider adding a sticky 'spots remaining' micro-banner or progress bar once "
            "users scroll past the hero, so the scarcity signal stays visible throughout "
            "the longer scroll journey to the registration form.",
        ],
    },
    {
        "sus_answers": [3, 2, 4, 1, 4, 2, 5, 2, 4, 1],
        "axe_penalty": 24.0,
        "feedback": (
            "The landing page is visually polished and well-structured for its narrow "
            "purpose — a single-weekend cycling tour — with a clear hero, stat bar, and "
            "prominent CTA. Navigation is minimal and purposeful, making the site very "
            "easy to learn. The primary usability concern is the 8 color-contrast "
            "violations (axe-core), most visibly the small amber-on-dark and medium-gray-"
            "on-cream text combinations, which reduce readability for users with low "
            "vision. Overall this is a competent, above-average marketing page that would "
            "benefit from contrast fixes and slightly larger body text before it clears "
            "a formal accessibility audit."
        ),
        "suggestions": [
            "Fix the 8 color-contrast violations: the slate-600 (#4b5563) text on cream "
            "(#faf8f3) background falls below 4.5:1 — darken to slate-700 (#374151) or "
            "increase font size to ≥18px to meet WCAG AA.",
            "Add a mobile hamburger menu — the nav-links are only visible on wide "
            "viewports; at the screenshot's apparent ~400px width they are hidden with "
            "no visible toggle shown.",
            "Increase the stats bar stat-label font size from 0.75rem to at least "
            "0.8125rem (13px) to improve legibility of the uppercase tracking labels "
            "('DAYS', 'ELEV CLIMB', etc.).",
            "The urgency dot animation ('12 spots remaining') draws the eye but the "
            "sentence is easy to miss at 0.88rem in the hero. Consider bumping to 0.95rem "
            "and adding a subtle highlight background to make the scarcity signal more "
            "scannable.",
            "Add visible section landmarks (e.g., <main>, landmark roles) so screen-reader "
            "users and keyboard navigators can jump between sections without relying "
            "solely on the skip link and nav anchors.",
            "The hero price anchor ('From $449') uses a very small 0.88rem font below the "
            "CTA buttons; consider surfacing it more prominently or moving it closer to "
            "the primary 'Reserve Your Spot' button to reduce the distance between the "
            "value proposition and the price.",
        ],
    },
]


def _violations_matching_penalty(target_penalty: float) -> list[dict]:
    """Synthesize a violations list that yields approximately `target_penalty`.

    scorer.axe_penalty weights critical=5, serious=3, moderate=1.5, minor=0.5
    per node. To get a specific total we emit serious (weight 3) violations
    until we're close, then pad with a single fractional minor.
    The penalty is capped at 30 in scorer.
    """
    violations: list[dict] = []
    remaining = min(target_penalty, 30.0)
    # Serious @ 3.0/node. Use int() so we undershoot then fine-tune.
    n_serious_nodes = int(remaining // 3.0)
    if n_serious_nodes:
        violations.append({
            "id": "demo-serious",
            "impact": "serious",
            "help": "Preserved from the real run's axe-core report.",
            "nodes": [{"html": "<span />"} for _ in range(n_serious_nodes)],
        })
        remaining -= n_serious_nodes * 3.0
    # Fill the gap with minor nodes @ 0.5/node.
    n_minor = int(round(remaining / 0.5))
    if n_minor > 0:
        violations.append({
            "id": "demo-minor",
            "impact": "minor",
            "help": "Padding to match recorded penalty.",
            "nodes": [{"html": "<em />"} for _ in range(n_minor)],
        })
    return violations


def seed_demo(runs_dir: Path) -> int:
    """Create one demo run with four iterations from the bundled real artifacts."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    store = storage.Storage(runs_dir / "design-gan.sqlite")
    run_id = store.create_run(BRIEF, "demo-seed")
    run_dir = runs_dir / f"run_{run_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    from .scorer import score as score_fn

    best_iter, best_score = 0, -1.0
    total_cost = 0.0
    for i, it in enumerate(_ITERS, start=1):
        iter_dir = run_dir / f"iter_{i:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Copy bundled site.html and screenshot.png into the run directory
        # so the viewer's file routes serve them identically to a real run.
        (iter_dir / "site.html").write_text(
            (_ASSETS / f"iter_{i}.html").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (iter_dir / "screenshot.png").write_bytes(
            (_ASSETS / f"iter_{i}.png").read_bytes()
        )

        violations = _violations_matching_penalty(it["axe_penalty"])
        result = score_fn(it["sus_answers"], violations)

        store.save_iteration(
            storage.IterationRecord(
                run_id=run_id,
                iter=i,
                html=(iter_dir / "site.html").read_text(encoding="utf-8"),
                sus_score=result.sus,
                axe_penalty=result.axe_penalty,
                composite_score=result.composite,
                sus_answers=it["sus_answers"],
                feedback=it["feedback"],
                suggestions=it["suggestions"],
                artifacts_dir=str(iter_dir),
                cost_usd=0.0,
            )
        )
        if result.composite > best_score:
            best_score = result.composite
            best_iter = i

    store.finish_run(run_id, best_iter, best_score, "converged")
    return run_id
