"""
Generate synthetic promotional/ad frames using PIL.

Produces a diverse set of ad-frame archetypes:
  - Brand logo with slogan
  - QR code with URL text
  - Product image placeholder with price tag
  - Text-only promo ("Use code SAVE20 for 20% off!")
  - Banner with gradient and CTA button

Run: python scripts/generate_ad_frames.py --output data/ad_frames --count 100
"""

import argparse
import os
import random
import math

from PIL import Image, ImageDraw, ImageFont

# Common color palettes for ad aesthetics
PALETTES = [
    [(255, 69, 0), (255, 215, 0)],       # Red-orange / gold
    [(30, 144, 255), (0, 255, 127)],     # Blue / green
    [(148, 0, 211), (255, 20, 147)],     # Purple / pink
    [(255, 140, 0), (255, 255, 255)],    # Orange / white
    [(0, 0, 0), (255, 255, 255)],        # Black / white (minimal)
    [(220, 20, 60), (255, 255, 255)],    # Crimson / white
]

BRANDS = [
    "NOVA BRAND", "PIXEL SHOP", "ULTRA PRO", "SPARK DEALS",
    "ZEST MARKET", "BUZZ ZONE", "VOLT STORE", "APEX GOODS",
    "PRIME PICK", "FLASH SALE", "TOP GEAR", "MEGA DEAL",
]

SLOGANS = [
    "Just Buy It.", "Taste the Savings.", "Experience More.",
    "Your Best Choice.", "Deals You Love.", "Why Pay More?",
    "Premium Quality.", "Limited Time Only!", "Act Now!",
]

CTAS = [
    "SHOP NOW", "BUY TODAY", "GET 50% OFF", "CLAIM DEAL",
    "ORDER NOW", "VISIT US", "SCAN HERE", "USE CODE: SAVE20",
]

PROMO_TEXTS = [
    "Use code SAVE20 for 20% off your next order!",
    "Follow us @brand for exclusive deals!",
    "Scan QR code to claim your FREE gift!",
    "Limited time offer — ends tonight!",
    "1M+ satisfied customers. Join today!",
    "Flash sale: 70% off selected items!",
    "Subscribe now and get a free month!",
    "Exclusive discount for our viewers only!",
]


def _get_font(size: int):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _gradient_bg(w: int, h: int, c1: tuple, c2: tuple, angle: float = 0.0) -> Image.Image:
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            t = (x * math.cos(angle) + y * math.sin(angle)) / (w + h)
            r = int(c1[0] + (c2[0] - c1[0]) * t)
            g = int(c1[1] + (c2[1] - c1[1]) * t)
            b = int(c1[2] + (c2[2] - c1[2]) * t)
            px[x, y] = (r, g, b)
    return img


def _draw_centered_text(draw, text, font, y, img_w, color):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img_w - tw) // 2, y), text, font=font, fill=color)


def gen_logo_banner(w=1280, h=720) -> Image.Image:
    palette = random.choice(PALETTES)
    img = _gradient_bg(w, h, palette[0], palette[1], angle=random.uniform(0, math.pi))
    draw = ImageDraw.Draw(img)

    brand = random.choice(BRANDS)
    slogan = random.choice(SLOGANS)
    cta = random.choice(CTAS)

    # Large brand name
    font_brand = _get_font(min(w // 8, 120))
    font_slogan = _get_font(min(w // 20, 48))
    font_cta = _get_font(min(w // 18, 52))

    text_color = palette[1] if sum(palette[0]) < 400 else palette[0]
    _draw_centered_text(draw, brand, font_brand, h // 4, w, text_color)
    _draw_centered_text(draw, slogan, font_slogan, h // 2, w, text_color)

    # CTA button
    btn_w, btn_h = w // 4, h // 8
    btn_x, btn_y = (w - btn_w) // 2, int(h * 0.68)
    draw.rounded_rectangle(
        [btn_x, btn_y, btn_x + btn_w, btn_y + btn_h],
        radius=btn_h // 3,
        fill=text_color,
    )
    btn_text_color = palette[0]
    _draw_centered_text(draw, cta, font_cta, btn_y + btn_h // 4, w, btn_text_color)

    return img


def gen_qr_placeholder(w=1280, h=720) -> Image.Image:
    """Generate a frame with a QR code placeholder (grid pattern) and URL text."""
    bg = (240, 240, 240) if random.random() > 0.5 else (20, 20, 20)
    fg = (20, 20, 20) if bg == (240, 240, 240) else (240, 240, 240)
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    # Fake QR code (grid of squares)
    qr_size = min(w, h) // 2
    qr_x = (w - qr_size) // 2
    qr_y = (h - qr_size) // 2 - 40
    cell = qr_size // 21

    random.seed(random.randint(0, 9999))
    for row in range(21):
        for col in range(21):
            if random.random() > 0.5 or (row < 7 and col < 7) or \
               (row < 7 and col > 13) or (row > 13 and col < 7):
                x0 = qr_x + col * cell
                y0 = qr_y + row * cell
                draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=fg)

    font = _get_font(min(w // 25, 44))
    url = f"www.{random.choice(BRANDS).lower().replace(' ', '')}.com"
    _draw_centered_text(draw, url, font, qr_y + qr_size + 20, w, fg)
    _draw_centered_text(draw, "SCAN TO VISIT", _get_font(min(w // 30, 36)), qr_y + qr_size + 60, w, fg)
    return img


def gen_text_promo(w=1280, h=720) -> Image.Image:
    palette = random.choice(PALETTES)
    img = _gradient_bg(w, h, palette[0], palette[1])
    draw = ImageDraw.Draw(img)
    text = random.choice(PROMO_TEXTS)
    text_color = (255, 255, 255)

    font_lg = _get_font(min(w // 15, 72))
    font_sm = _get_font(min(w // 25, 44))
    brand = random.choice(BRANDS)
    _draw_centered_text(draw, brand, font_lg, h // 3, w, text_color)
    _draw_centered_text(draw, text, font_sm, h // 2, w, text_color)
    return img


def gen_product_placeholder(w=1280, h=720) -> Image.Image:
    bg = (245, 245, 245)
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    # Product box placeholder
    box_w, box_h = w // 3, int(h * 0.6)
    box_x, box_y = (w - box_w) // 2, h // 8
    draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=(200, 200, 200), outline=(100, 100, 100), width=3)
    draw.text((box_x + 20, box_y + box_h // 2 - 20), "[PRODUCT IMAGE]", font=_get_font(28), fill=(100, 100, 100))

    # Price tag
    price = f"${random.randint(9, 299)}.{random.choice(['99', '00', '95'])}"
    tag_font = _get_font(min(w // 12, 88))
    price_color = (220, 20, 60)
    _draw_centered_text(draw, price, tag_font, box_y + box_h + 20, w, price_color)

    brand = random.choice(BRANDS)
    _draw_centered_text(draw, brand, _get_font(min(w // 22, 52)), 30, w, (50, 50, 50))
    return img


GENERATORS = [gen_logo_banner, gen_qr_placeholder, gen_text_promo, gen_product_placeholder]


def generate_ad_frames(output_dir: str, count: int = 100, width: int = 1280, height: int = 720):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(count):
        gen_fn = GENERATORS[i % len(GENERATORS)]
        img = gen_fn(w=width, h=height)
        path = os.path.join(output_dir, f"ad_{i:04d}.jpg")
        img.save(path, quality=92)
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{count}")
    print(f"Saved {count} ad frames to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/ad_frames")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    generate_ad_frames(args.output, args.count, args.width, args.height)
