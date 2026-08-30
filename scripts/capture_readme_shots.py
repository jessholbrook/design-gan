"""Capture documentation screenshots from a running viewer."""

import os
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
BASE = os.environ.get("DESIGN_GAN_SCREENSHOT_BASE", "http://127.0.0.1:8000")
RUN_ID = os.environ.get("DESIGN_GAN_SCREENSHOT_RUN_ID", "2")


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page = ctx.new_page()

        # 1. Scrubber, single mode, latest iteration
        page.goto(f"{BASE}/runs/{RUN_ID}/scrub", wait_until="networkidle")
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

        # Capture the latest available iteration.
        max_idx = int(page.locator("#scrub-range").get_attribute("max") or "0")
        seek(max_idx)
        page.screenshot(path=str(OUT / "scrubber-single.png"), full_page=False)

        # 2. Compare latest candidate against its parent.
        page.click("button:has-text('vs prev')")
        page.wait_for_timeout(300)
        page.screenshot(path=str(OUT / "scrubber-compare.png"), full_page=False)

        # 3. Run page (the per-iteration cards landing view)
        page.goto(f"{BASE}/runs/{RUN_ID}", wait_until="networkidle")
        page.screenshot(path=str(OUT / "run-page.png"), full_page=False)

        browser.close()
    print("wrote:", *sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
