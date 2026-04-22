"""Seed a fake run with pre-baked iterations so the viewer has something to show
without requiring an API key or Playwright install."""

from __future__ import annotations

import struct
import time
import zlib
from pathlib import Path

from . import storage

# 4 sample iterations with progressively better SUS scores and feedback.
_ITERS = [
    {
        "html": None,
        "sus_answers": [2, 4, 2, 4, 2, 4, 2, 4, 2, 4],  # SUS = 0
        "feedback": (
            "The page is a wall of unstyled text. The headline is lost, the call-to-action "
            "blends into the paragraph, and there is no visual hierarchy whatsoever."
        ),
        "suggestions": [
            "Add a distinct hero section with a primary headline.",
            "Promote the CTA to a visible button.",
            "Introduce a typographic scale (h1/h2/body).",
            "Apply a consistent color palette.",
        ],
        "axe_count": 6,
        "rgb": (232, 150, 150),
    },
    {
        "sus_answers": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # SUS = 50
        "feedback": (
            "A hero section and a visible CTA now exist, but spacing is cramped and "
            "contrast on the CTA is borderline. Information density is still high."
        ),
        "suggestions": [
            "Increase vertical rhythm between sections.",
            "Raise CTA contrast to at least 4.5:1.",
            "Break the single-column layout into scannable chunks.",
        ],
        "axe_count": 3,
        "rgb": (230, 200, 130),
    },
    {
        "sus_answers": [4, 2, 4, 2, 4, 2, 4, 2, 4, 2],  # SUS = 75
        "feedback": (
            "Layout is clean and the CTA is unambiguous. Typography is coherent, and "
            "the site communicates its purpose within a glance. A11y is solid."
        ),
        "suggestions": [
            "Add supporting social-proof or testimonials below the fold.",
            "Consider an accent color for interactive elements.",
        ],
        "axe_count": 1,
        "rgb": (170, 210, 160),
    },
    {
        "sus_answers": [5, 1, 5, 1, 4, 2, 5, 1, 5, 1],  # SUS = 90
        "feedback": (
            "Excellent. The page now reads at a glance, hierarchy guides the eye to the "
            "CTA, and supporting content reinforces the primary action. No a11y issues."
        ),
        "suggestions": [
            "Optional: tighten copy in the second section for mobile.",
        ],
        "axe_count": 0,
        "rgb": (120, 200, 140),
    },
]


# Increasingly polished HTML samples.
_HTML_SAMPLES = [
    # Iter 1: unstyled
    """<!doctype html><html><body>
<p>Weekend cycling tour in rural Vermont. October foliage, quiet back roads, farm-to-table
stops. Small groups. Book now by emailing tours@example.com.</p>
</body></html>""",
    # Iter 2: basic hero + CTA
    """<!doctype html><html><body style="font-family:sans-serif;max-width:640px;margin:40px auto">
<h1>Vermont Cycling Tour</h1>
<p>A weekend of back-road cycling through rural Vermont during peak foliage. Small groups,
farm-to-table meals, comfortable inns.</p>
<a href="#book" style="background:#ccc;padding:8px 16px">Book a spot</a>
</body></html>""",
    # Iter 3: real layout
    """<!doctype html><html><body style="font-family:system-ui;margin:0;color:#222">
<header style="padding:64px 24px;background:#fafaf8;text-align:center">
  <h1 style="font-size:40px;margin:0 0 12px">Ride Vermont in October</h1>
  <p style="max-width:560px;margin:0 auto 24px;color:#555">A 3-day cycling weekend through
     rural Vermont — quiet roads, peak foliage, farm-to-table meals, small groups.</p>
  <a href="#book" style="background:#2a6;color:#fff;padding:12px 24px;border-radius:6px;
     text-decoration:none;font-weight:600">Book your weekend</a>
</header>
<section style="padding:48px 24px;max-width:720px;margin:0 auto">
  <h2>What's included</h2>
  <ul><li>3 days of guided riding</li><li>All meals</li><li>Inn lodging</li></ul>
</section>
</body></html>""",
    # Iter 4: polished
    """<!doctype html><html><body style="font-family:Georgia,serif;margin:0;color:#1a1a1a;
     background:#fdfdfa">
<header style="padding:96px 32px;text-align:center;border-bottom:1px solid #eee">
  <p style="letter-spacing:.2em;color:#888;font-size:12px;margin:0">OCTOBER 11-13</p>
  <h1 style="font-size:48px;margin:12px 0 16px;font-weight:400">Three days on<br>Vermont's quietest roads</h1>
  <p style="max-width:560px;margin:0 auto 32px;color:#444;font-size:18px;line-height:1.6">
     A small-group cycling weekend through the Green Mountains at peak foliage. Farm-to-table
     dinners. Handpicked inns. 35 miles a day.</p>
  <a href="#book" style="background:#1a1a1a;color:#fff;padding:14px 32px;border-radius:4px;
     text-decoration:none;font-weight:600;letter-spacing:.05em">RESERVE YOUR SPOT</a>
</header>
<main style="max-width:720px;margin:64px auto;padding:0 32px;line-height:1.7">
  <h2 style="font-weight:400">The weekend</h2>
  <p>You'll ride with 6 other cyclists and one guide. The route loops from Woodstock through
     Barnard and the Pomfret hollows — dirt roads, covered bridges, no traffic.</p>
  <h2 style="font-weight:400;margin-top:48px">What's included</h2>
  <p>Two nights at the Barnard Inn, five meals, a support van, and a route sheet you'll never
     need to open.</p>
</main>
</body></html>""",
]

for i, it in enumerate(_ITERS):
    it["html"] = _HTML_SAMPLES[i]


def _make_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Tiny pure-stdlib PNG writer for a solid-color image."""
    r, g, b = rgb
    raw = bytearray()
    for _ in range(height):
        raw.append(0)  # filter type: None
        for _ in range(width):
            raw.extend([r, g, b])

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
    iend = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def _fake_violation(i: int) -> dict:
    impacts = ["critical", "serious", "moderate", "minor"]
    return {
        "id": f"demo-{i}",
        "impact": impacts[i % 4],
        "help": "Placeholder a11y issue for demo data.",
        "nodes": [{"html": "<span />"}],
    }


def seed_demo(runs_dir: Path) -> int:
    """Create one fake run with 4 iterations. Returns the run id."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    store = storage.Storage(runs_dir / "design-gan.sqlite")
    run_id = store.create_run(
        "DEMO: A landing page for a weekend cycling tour in rural Vermont.",
        "demo-seed",
    )
    run_dir = runs_dir / f"run_{run_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_iter, best_score = 0, -1.0
    for i, it in enumerate(_ITERS, start=1):
        iter_dir = run_dir / f"iter_{i:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "site.html").write_text(it["html"], encoding="utf-8")
        (iter_dir / "screenshot.png").write_bytes(_make_png(320, 180, it["rgb"]))

        # Scoring
        from .scorer import score as score_fn

        violations = [_fake_violation(j) for j in range(it["axe_count"])]
        result = score_fn(it["sus_answers"], violations)

        store.save_iteration(
            storage.IterationRecord(
                run_id=run_id,
                iter=i,
                html=it["html"],
                sus_score=result.sus,
                axe_penalty=result.axe_penalty,
                composite_score=result.composite,
                sus_answers=it["sus_answers"],
                feedback=it["feedback"],
                suggestions=it["suggestions"],
                artifacts_dir=str(iter_dir),
            )
        )
        if result.composite > best_score:
            best_score = result.composite
            best_iter = i
        # Space iterations out a bit so the viewer's timestamps look plausible.
        time.sleep(0.01)

    store.finish_run(run_id, best_iter, best_score, "converged")
    return run_id
