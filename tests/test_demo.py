"""Tests for demo.seed_demo — the fixture-based run that needs no API key."""

from __future__ import annotations

from pathlib import Path

from design_gan.demo import seed_demo
from design_gan.storage import Storage


def test_seed_demo_creates_run_with_four_iterations(tmp_path: Path):
    run_id = seed_demo(tmp_path)
    assert run_id >= 1

    store = Storage(tmp_path / "design-gan.sqlite")
    run = store.get_run(run_id)
    assert run is not None
    assert run["status"] == "converged"

    iters = store.iterations_for_run(run_id)
    assert len(iters) == 4

    # Real run peaked at iter 3 then regressed, so composites are NOT
    # monotonically increasing. But the peak should still beat the start.
    composites = [it["composite_score"] for it in iters]
    assert max(composites) > composites[0]


def test_seed_demo_writes_artifacts_to_disk(tmp_path: Path):
    run_id = seed_demo(tmp_path)
    run_dir = tmp_path / f"run_{run_id:04d}"
    assert run_dir.is_dir()
    for i in range(1, 5):
        it_dir = run_dir / f"iter_{i:03d}"
        assert (it_dir / "site.html").is_file()
        assert (it_dir / "screenshot.png").is_file()
        assert (it_dir / "screenshot.png").stat().st_size > 0


def test_seed_demo_best_iter_matches_highest_composite(tmp_path: Path):
    run_id = seed_demo(tmp_path)
    store = Storage(tmp_path / "design-gan.sqlite")
    run = store.get_run(run_id)
    iters = store.iterations_for_run(run_id)
    best = max(iters, key=lambda it: it["composite_score"])
    assert run["best_iter"] == best["iter"]
    assert run["best_score"] == best["composite_score"]


def test_seed_demo_is_reentrant(tmp_path: Path):
    # Running it twice in the same dir must not collide and produce two distinct runs.
    r1 = seed_demo(tmp_path)
    r2 = seed_demo(tmp_path)
    assert r1 != r2

    store = Storage(tmp_path / "design-gan.sqlite")
    assert len(store.list_runs()) == 2


def test_seed_demo_generates_valid_png(tmp_path: Path):
    run_id = seed_demo(tmp_path)
    png = (tmp_path / f"run_{run_id:04d}" / "iter_001" / "screenshot.png").read_bytes()
    # PNG magic bytes.
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
