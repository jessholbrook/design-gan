"""Renderer: headless browser snapshot of a generated site."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Playwright is a heavy dependency (ships a browser). We import it lazily inside
# render() so the rest of the package (storage, scorer, viewer) can be used
# without it installed.

# Pinned to a stable release.
AXE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.0/axe.min.js"


@dataclass
class RenderResult:
    screenshot_png: bytes
    dom_html: str
    axe_violations: list[dict[str, Any]] = field(default_factory=list)
    axe_error: str | None = None  # set when axe-core fails to load/run
    console_errors: list[str] = field(default_factory=list)
    viewport: tuple[int, int] = (1280, 800)

    @property
    def screenshot_b64(self) -> str:
        return base64.standard_b64encode(self.screenshot_png).decode("ascii")


async def render(html: str, viewport: tuple[int, int] = (1280, 800)) -> RenderResult:
    """Render HTML in headless Chromium and capture screenshot + DOM + a11y report."""
    from playwright.async_api import async_playwright

    console_errors: list[str] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            context = await browser.new_context(
                viewport={"width": viewport[0], "height": viewport[1]},
            )
            page = await context.new_page()
            page.on(
                "console",
                lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
            )

            await page.set_content(html, wait_until="domcontentloaded")
            # Settle any post-load scripts.
            await page.wait_for_timeout(300)

            screenshot = await page.screenshot(full_page=False, type="png")
            dom_html = await page.content()

            axe_violations: list[dict[str, Any]] = []
            axe_error: str | None = None
            try:
                await page.add_script_tag(url=AXE_CDN)
                result_json = await page.evaluate(
                    "async () => JSON.stringify(await axe.run(document))"
                )
                parsed = json.loads(result_json)
                axe_violations = parsed.get("violations", [])
            except Exception as e:
                # axe-core unavailable (offline, blocked). Record the reason so
                # callers can tell "no violations" from "never ran".
                axe_error = f"{type(e).__name__}: {e}"

            return RenderResult(
                screenshot_png=screenshot,
                dom_html=dom_html,
                axe_violations=axe_violations,
                axe_error=axe_error,
                console_errors=console_errors,
                viewport=viewport,
            )
        finally:
            await browser.close()


def write_artifacts(result: RenderResult, out_dir: Path) -> dict[str, Path]:
    """Persist render artifacts to disk. Returns a map of kind -> path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shot = out_dir / "screenshot.png"
    dom = out_dir / "dom.html"
    axe = out_dir / "axe.json"
    shot.write_bytes(result.screenshot_png)
    dom.write_text(result.dom_html, encoding="utf-8")
    axe.write_text(
        json.dumps(
            {
                "violations": result.axe_violations,
                "axe_error": result.axe_error,
                "console_errors": result.console_errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"screenshot": shot, "dom": dom, "axe": axe}
