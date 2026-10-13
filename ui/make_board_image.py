"""Build ui/static/board.png from the "Board Image.png" screenshot.

Crops the board spaces out of the screenshot and wipes the game state that was
captured with it (High Council seats, Spies, control banners, the Conflict
panel), so the UI can draw the live state on top.

    python ui/make_board_image.py
"""
import os

import numpy as np

from PIL import Image, ImageDraw, ImageFilter, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "Board Image.png")
OUT = os.path.join(HERE, "static", "board.png")
CROP = (412, 48, 1680, 843)        # keep in sync with BOARD_W/BOARD_H in app.js


def copy_disc(im, src, dst, r):
    """Paste a disc of radius r centred on src over dst."""
    patch = im.crop((src[0] - r, src[1] - r, src[0] + r, src[1] + r))
    mask = Image.new("L", patch.size, 0)
    ImageDraw.Draw(mask).ellipse((0, 0, 2 * r - 1, 2 * r - 1), fill=255)
    im.paste(patch, (dst[0] - r, dst[1] - r), mask.filter(ImageFilter.GaussianBlur(1)))


def grey_out(im, box):
    region = im.crop(box)
    g = ImageOps.grayscale(region).point(lambda v: int(v * 0.55))
    im.paste(Image.merge("RGB", (g, g, g)), box)


def grey_green(im, box):
    """Recolor only the green-tinted pixels in box (a Spy's connector lines)."""
    a = np.asarray(im.crop(box)).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = (g > r + 25) & (g > b + 5)
    lum = np.clip((r + g + b) / 3 * 1.6, 0, 255)
    a[m] = lum[m][:, None]
    im.paste(Image.fromarray(a.astype(np.uint8)), box)


def main():
    im = Image.open(SRC).convert("RGB")
    # High Council seats taken by red and blue -> empty seat
    for x in (796, 888):
        copy_disc(im, (752, 96), (x, 96), 20)
    # green Spies on the Emperor and Guild posts, blue on the Landsraad and
    # Deep Desert posts -> empty post
    for y in (148, 349):
        grey_green(im, (628, y - 55, 666, y + 90))          # green connector lines
        copy_disc(im, (647, 547), (647, y), 18)
    copy_disc(im, (1117, 157), (1030, 141), 18)
    copy_disc(im, (1387, 578), (1115, 578), 18)
    # red marker on the Research Station corner -> same corner of Sietch Tabr
    im.paste(im.crop((930, 600, 975, 640)), (930, 457))
    # control banners (Spice Refinery, Arrakeen, Imperial Basin) -> neutral
    for x in (850, 1105, 1365):
        grey_out(im, (x, 323, x + 32, 356))
    im = im.crop(CROP)
    # the captured Conflict panel -> dark plate (the live Conflict is drawn there)
    d = ImageDraw.Draw(im, "RGBA")
    x0, y0 = 1112 - CROP[0], 606 - CROP[1]
    d.rounded_rectangle((x0, y0, CROP[2] - CROP[0] - 4, CROP[3] - CROP[1] - 2),
                        radius=10, fill=(20, 16, 12, 255))
    im.save(OUT, optimize=True)
    print("wrote", OUT, im.size)


if __name__ == "__main__":
    main()
