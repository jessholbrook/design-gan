"""One-off: capture README screenshots of the scrubber UI.

Assumes the viewer is running at http://127.0.0.1:8001 against runs_real
(see .claude/launch.json — `viewer-real`).
"""
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://127.0.0.1:8001"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page = ctx.new_page()

        # 1. Dashboard
        page.goto(f"{BASE}/", wait_until="networkidle")
        page.screenshot(path=str(OUT / "dashboard.png"), full_page=False)

        # 2. Scrubber, single mode, mid-run iteration
        page.goto(f"{BASE}/runs/1/scrub", wait_until="networkidle")
        page.wait_for_selector("#scrub-range")

        def seek(idx0: int) -> None:
            page.evaluate(
                "(v) => { const s = document.getElementById('scrub-range');"
                " s.value = String(v);"
                " s.dispatchEvent(new Event('input', {bubbles:true}));"
                " s.dispatchEvent(new Event('change', {bubbles:true})); }",
                idx0,
            )
            page.wait_for_timeout(400)

        # Seek to iteration 3 (zero-indexed value 2) — a representative mid-run state
        seek(2)
        page.screenshot(path=str(OUT / "scrubber-single.png"), full_page=False)

        # 3. Compare mode vs best — overlay the draggable divider
        page.click("button:has-text('vs best')")
        page.wait_for_timeout(300)
        # Seek to iteration 1 (zero-indexed value 0) so the diff against the best is dramatic
        seek(0)
        page.screenshot(path=str(OUT / "scrubber-compare.png"), full_page=False)

        # 4. Run page (the per-iteration cards landing view)
        page.goto(f"{BASE}/runs/1", wait_until="networkidle")
        page.screenshot(path=str(OUT / "run-page.png"), full_page=False)

        browser.close()
    print("wrote:", *sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
