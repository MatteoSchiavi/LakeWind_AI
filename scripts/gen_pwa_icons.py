"""Generate LakeWind PWA icons (Phase 4 / W4).

Simple, on-brand: rounded deep-navy square, white wave glyph, wind-speed
green dot accent. Generated with Pillow (already a runtime dependency) so
no binary asset has to be hand-painted or vendored.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).resolve().parent.parent / "web-ui" / "public" / "icons"

BG = (11, 18, 32)        # #0b1220 deep navy
WAVE = (56, 189, 248)    # #38bdf8 sky
WAVE2 = (2, 132, 199)    # #0284c7 sky-600
DOT = (34, 197, 94)      # #22c55e sailable green


def _rounded_bg(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=size // 5, fill=BG)
    return img


def _draw_wave(d: ImageDraw.Draw, size: int, y: float, amp: float, color, width: int) -> None:
    """One sine-like wave across the icon, drawn as connected line segments."""
    pts = []
    steps = 48
    x0, x1 = size * 0.14, size * 0.86
    for i in range(steps + 1):
        t = i / steps
        x = x0 + (x1 - x0) * t
        yy = y + amp * (2.2 * t - 1) ** 2  # parabola approximating a swell
        pts.append((x, yy))
    d.line(pts, fill=color, width=width, joint="curve")


def make_icon(size: int) -> Image.Image:
    img = _rounded_bg(size)
    d = ImageDraw.Draw(img)
    # Two swells
    _draw_wave(d, size, size * 0.56, size * 0.10, WAVE2, max(6, size // 42))
    _draw_wave(d, size, size * 0.68, size * 0.10, WAVE, max(6, size // 42))
    # Wind dot (sailable green) — top right
    r = size * 0.07
    cx, cy = size * 0.70, size * 0.30
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=DOT)
    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for size in (192, 512):
        make_icon(size).save(OUT_DIR / f"icon-{size}.png", "PNG")
        print(f"wrote {OUT_DIR / f'icon-{size}.png'}")


if __name__ == "__main__":
    main()
