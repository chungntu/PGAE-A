# =============================================================================
# Renders the higher-resolution figures (fig4, fig6, fig7, fig8, fig9) from
# their HTML sources in code/figures/ to PNG in the same folder.
# =============================================================================
import os
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
TARGETS = ["fig1.html", "fig4.html", "fig6.html", "fig7.html", "fig8.html", "fig9.html", "fig10.html"]


async def export_all():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # device_scale_factor must match the DPR used inside each HTML's canvas
        # backing-store scaling (const DPR=4), so the screenshot actually
        # captures the canvas's full internal resolution instead of
        # downsampling it back to CSS size.
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080}, device_scale_factor=4
        )
        page = await context.new_page()
        for f in TARGETS:
            src = FIG_DIR / f
            if not src.exists():
                print(f"MISSING {f}, skipping")
                continue
            print(f"Converting {f}...")
            file_url = f"file:///{str(src).replace(os.sep, '/')}"
            out_path = FIG_DIR / f.replace(".html", ".png")
            await page.goto(file_url, wait_until="networkidle")
            svg = await page.query_selector("svg")
            canvas = await page.query_selector("canvas")
            if svg:
                await svg.screenshot(path=str(out_path))
            elif canvas:
                await canvas.screenshot(path=str(out_path))
            else:
                await page.screenshot(path=str(out_path), full_page=True)
        await context.close()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(export_all())
    print("done")
