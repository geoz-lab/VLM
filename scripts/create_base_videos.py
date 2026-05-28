"""
Generate a diverse set of short base videos for synthetic dataset creation.

Creates 10 videos with different visual characteristics:
  - Talking-head (static face region, subtitle-like bottom bar)
  - Nature documentary (slow color gradient)
  - Sports (high-motion simulation)
  - Tutorial / screen recording (plain background + text)
  - News broadcast (split frame with text ticker)

Run: python scripts/create_base_videos.py --output data/raw_videos --duration 20
"""

import argparse
import os
import math
import random

import cv2
import numpy as np


W, H = 640, 360
FPS = 30


def _put(frame, text, x, y, scale=0.6, color=(255, 255, 255), thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def writer(path, fps=FPS, w=W, h=H):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


# ── Video generators ──────────────────────────────────────────────────────────

def gen_talking_head(path, duration=20):
    """Static background + person region + scrolling subtitle bar."""
    out = writer(path)
    bg = (45, 55, 72)  # dark blue-grey
    face_color = (180, 140, 110)
    scenes = [
        (0,      int(duration * 0.4), "Interview: talking about technology trends"),
        (int(duration * 0.4), int(duration * 0.7), "Discussing the latest research findings"),
        (int(duration * 0.7), duration, "Answering audience questions live"),
    ]
    for fidx in range(duration * FPS):
        t = fidx / FPS
        frame = np.full((H, W, 3), bg, dtype=np.uint8)
        # Face ellipse
        cv2.ellipse(frame, (W // 2, H // 3), (80, 90), 0, 0, 360, face_color, -1)
        # Shoulders
        cv2.ellipse(frame, (W // 2, H * 2 // 3), (120, 60), 0, 0, 180, face_color, -1)
        # Bottom subtitle bar
        cv2.rectangle(frame, (0, H - 40), (W, H), (20, 20, 20), -1)
        for s, e, text in scenes:
            if s <= t < e:
                _put(frame, text[:60], 10, H - 14, 0.45, (220, 220, 220))
                break
        _put(frame, f"LIVE", W - 60, 30, 0.5, (0, 0, 220), 2)
        out.write(frame)
    out.release()


def gen_nature_doc(path, duration=20):
    """Slow gradient color shifts — forest/sky/ocean scenes."""
    out = writer(path)
    palette = [
        ((34, 85, 34),   (70, 130, 70),  "Forest at dawn"),
        ((70, 130, 180), (135, 206, 250), "Clear sky"),
        ((0, 105, 148),  (0, 180, 220),  "Ocean surface"),
        ((139, 90, 43),  (180, 130, 70), "Desert dunes"),
    ]
    n = duration * FPS
    for fidx in range(n):
        t = fidx / FPS
        pi = int(t / duration * len(palette))
        pi = min(pi, len(palette) - 1)
        c1, c2, label = palette[pi]
        blend = (t * FPS % (duration * FPS / len(palette))) / (duration * FPS / len(palette))
        color = tuple(int(c1[i] + (c2[i] - c1[i]) * blend) for i in range(3))
        frame = np.full((H, W, 3), color, dtype=np.uint8)
        # Horizon line
        cv2.line(frame, (0, H * 2 // 3), (W, H * 2 // 3), (int(color[0] * 0.7), int(color[1] * 0.7), int(color[2] * 0.7)), 2)
        _put(frame, label, 20, H - 20, 0.5, (255, 255, 255))
        out.write(frame)
    out.release()


def gen_sports(path, duration=20):
    """High-motion: fast-moving elements + score overlay."""
    out = writer(path)
    bg = (34, 139, 34)  # grass green
    for fidx in range(duration * FPS):
        t = fidx / FPS
        frame = np.full((H, W, 3), bg, dtype=np.uint8)
        # Field lines
        cv2.line(frame, (W // 2, 0), (W // 2, H), (255, 255, 255), 2)
        cv2.circle(frame, (W // 2, H // 2), 60, (255, 255, 255), 2)
        # Moving ball
        bx = int(W // 2 + math.sin(t * 2) * 200)
        by = int(H // 2 + math.cos(t * 3) * 100)
        cv2.circle(frame, (bx, by), 12, (255, 255, 255), -1)
        # Scoreboard
        cv2.rectangle(frame, (0, 0), (200, 35), (0, 0, 0), -1)
        score_a = int(t / 5)
        score_b = int(t / 7)
        _put(frame, f"TEAM A {score_a} - {score_b} TEAM B", 8, 24, 0.5, (255, 255, 100))
        out.write(frame)
    out.release()


def gen_tutorial(path, duration=20):
    """Screen-recording style: light background, code-like text blocks."""
    out = writer(path)
    bg = (245, 245, 245)
    sections = [
        (0,   6,  "Step 1: Install the dependencies"),
        (6,   12, "Step 2: Configure the environment"),
        (12,  18, "Step 3: Run the main script"),
        (18,  20, "Step 4: Review the output"),
    ]
    code_lines = [
        "$ pip install -r requirements.txt",
        "$ python setup.py --config default",
        "$ python main.py --input data/",
        "  Processing... done (3.2s)",
        "$ cat results/output.json",
    ]
    for fidx in range(duration * FPS):
        t = fidx / FPS
        frame = np.full((H, W, 3), bg, dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (W, 40), (50, 50, 50), -1)
        _put(frame, "Tutorial — Video Ad Insertion Demo", 10, 26, 0.55, (200, 200, 200))
        for s, e, label in sections:
            if s <= t < e:
                _put(frame, label, 20, 70, 0.6, (30, 30, 30))
                break
        for i, line in enumerate(code_lines[:int(t / (duration / len(code_lines))) + 1]):
            _put(frame, line, 20, 110 + i * 28, 0.5, (0, 100, 0) if line.startswith("$") else (80, 80, 80))
        out.write(frame)
    out.release()


def gen_news(path, duration=20):
    """News broadcast: anchor desk + ticker."""
    out = writer(path)
    topics = [
        "BREAKING: Major climate summit reaches new agreement",
        "Markets: Tech stocks rally on strong earnings reports",
        "Science: New study shows surprising biodiversity data",
        "Politics: International talks yield positive results",
    ]
    for fidx in range(duration * FPS):
        t = fidx / FPS
        frame = np.full((H, W, 3), (20, 30, 60), dtype=np.uint8)
        # Anchor desk area
        cv2.rectangle(frame, (W // 4, H // 5), (W * 3 // 4, H * 4 // 5), (40, 60, 100), -1)
        # Anchor silhouette
        cv2.ellipse(frame, (W // 2, H // 3), (55, 60), 0, 0, 360, (160, 120, 90), -1)
        # Channel logo
        cv2.rectangle(frame, (W - 90, 5), (W - 5, 40), (200, 0, 0), -1)
        _put(frame, "NEWS", W - 82, 28, 0.6, (255, 255, 255), 2)
        # Ticker bar
        cv2.rectangle(frame, (0, H - 45), (W, H), (180, 0, 0), -1)
        topic = topics[int(t / (duration / len(topics))) % len(topics)]
        offset = int((t * 60) % (W + len(topic) * 10)) - len(topic) * 10
        _put(frame, topic, W - offset, H - 16, 0.48, (255, 255, 255))
        out.write(frame)
    out.release()


GENERATORS = [
    ("talking_head", gen_talking_head),
    ("nature_doc", gen_nature_doc),
    ("sports", gen_sports),
    ("tutorial", gen_tutorial),
    ("news", gen_news),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw_videos")
    parser.add_argument("--duration", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    for name, gen_fn in GENERATORS:
        path = os.path.join(args.output, f"{name}.mp4")
        print(f"  Generating {name}.mp4 ...")
        gen_fn(path, duration=args.duration)

    print(f"\nCreated {len(GENERATORS)} base videos in {args.output}/")


if __name__ == "__main__":
    main()
