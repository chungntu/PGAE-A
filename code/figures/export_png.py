# =============================================================================
# Author: Truong Thanh Chung (1,2)
#   (1) University of Transport and Communications, No. 3 Cau Giay Street,
#       Hanoi, Vietnam
#   (2) Nha Trang University, 02 Nguyen Dinh Chieu, Khanh Hoa, Vietnam
#
# Code and reproducible baselines for the PIAE-A study: Physics-Informed
# Autoencoders with Unsupervised Feature Attention for Label-Free Structural
# Damage Detection (Z24 bridge benchmark).
# =============================================================================

import os
import asyncio
from playwright.async_api import async_playwright

SCALE_FACTOR = 4

async def export_all():
    base_dir = os.path.dirname(os.path.abspath(__file__))   # code/figures
    png_dir = os.path.join(base_dir, "png")
    os.makedirs(png_dir, exist_ok=True)
    
    html_files = [f for f in os.listdir(base_dir) if f.endswith(".html")]
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Render at 4x CSS resolution so figures remain sharp in the manuscript.
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=SCALE_FACTOR,
        )
        page = await context.new_page()
        
        for f in html_files:
            print(f"Converting {f}...")
            file_url = f"file:///{os.path.join(base_dir, f).replace(chr(92), '/')}"
            png_name = f.replace(".html", ".png")
            out_path = os.path.join(png_dir, png_name)
            
            await page.goto(file_url, wait_until="networkidle")
            
            svg = await page.query_selector('svg')
            plotly = await page.query_selector('.js-plotly-plot')
            canvas = await page.query_selector('canvas')
            
            if plotly:
                await plotly.screenshot(path=out_path)
            elif svg:
                await svg.screenshot(path=out_path)
            elif canvas:
                await canvas.screenshot(path=out_path)
            else:
                # Try to get the first child of body to screenshot
                body_first_child = await page.query_selector('body > *:first-child')
                if body_first_child:
                    await body_first_child.screenshot(path=out_path)
                else:
                    await page.screenshot(path=out_path, full_page=True)
                
        await context.close()
        await browser.close()

if __name__ == "__main__":
    asyncio.run(export_all())
