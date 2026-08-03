"""Dev-only script: capture dashboard states and render docs/demo.gif.

Requires: uv pip install playwright && playwright install chromium

Usage: .venv/bin/python scripts/make_demo_gif.py [--base http://127.0.0.1:8000]
Frames are written to /tmp/mulehunt_frames/, then assembled with ffmpeg.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

FRAME_DIR = Path("/tmp/mulehunt_frames")
OUT = Path("docs/demo.gif")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1000)
    args = ap.parse_args()

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for f in FRAME_DIR.glob("*.png"):
        f.unlink()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.goto(args.base, wait_until="networkidle")

        page.wait_for_selector("#top-table tbody tr", timeout=20000)
        page.wait_for_selector("#ring-svg")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(FRAME_DIR / "1_overview.png"))

        page.click("#top-table tbody tr")
        page.wait_for_selector("#detail h3", timeout=10000)
        page.wait_for_timeout(800)
        page.screenshot(path=str(FRAME_DIR / "2_account.png"))

        page.click("#explain-btn")
        page.wait_for_function(
            "document.getElementById('explain-text').textContent.startsWith('generating') === false && "
            "document.getElementById('explain-text').textContent.length > 10",
            timeout=60000,
        )
        page.wait_for_timeout(1200)
        page.screenshot(path=str(FRAME_DIR / "3_explain.png"))

        page.select_option("#ring-select", index=0)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(FRAME_DIR / "4_ring.png"))

        browser.close()

    frames = ["1_overview", "2_account", "3_explain", "4_ring"]
    hold = "1.6"
    inputs = []
    for f in frames:
        inputs += ["-loop", "1", "-t", hold, "-i", str(FRAME_DIR / f"{f}.png")]
    fc = ";".join(f"[{i}:v]scale=900:-1,split=1[a{i}]" for i in range(len(frames)))
    fc += ";" + "".join(f"[a{i}]" for i in range(len(frames))) + f"concat=n={len(frames)}:v=1:a=0[out]"
    subprocess.run(
        ["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[out]", str(OUT)],
        check=True,
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
