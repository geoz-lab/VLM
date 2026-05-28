"""
Create a short synthetic test video (30s, 30fps) with simple colored scenes.
No external footage needed.
"""
import os
import cv2
import numpy as np

def create_test_video(output_path: str, duration_s: int = 30, fps: int = 30, w: int = 640, h: int = 360):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    total = duration_s * fps

    # Scene definitions: (start_frac, end_frac, base_color, label)
    scenes = [
        (0.00, 0.30, (70, 130, 180), "Sky blue — interview scene"),
        (0.30, 0.55, (34,  85,  34), "Forest green — nature scene"),
        (0.55, 0.80, (180,  60,  60), "Red — action scene"),
        (0.80, 1.00, (100,  60, 160), "Purple — closing scene"),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX

    for fidx in range(total):
        t = fidx / fps
        frac = fidx / total

        # Pick scene
        scene_label = ""
        base_color = (128, 128, 128)
        for s_start, s_end, color, label in scenes:
            if s_start <= frac < s_end:
                base_color = color
                scene_label = label
                break

        # Very slight brightness flicker (tiny noise, realistic but low)
        noise = int(np.random.uniform(-2, 2))
        color = tuple(max(0, min(255, c + noise)) for c in base_color)
        frame = np.full((h, w, 3), color, dtype=np.uint8)

        # Draw static scene text (no animation — keeps per-frame pHash stable)
        cv2.putText(frame, scene_label, (20, 40), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Scene content", (20, h - 20), font, 0.5, (200, 200, 200), 1)

        out.write(frame)

    out.release()
    print(f"Created test video: {output_path}  ({duration_s}s @ {fps}fps, {w}x{h})")

if __name__ == "__main__":
    create_test_video("data/raw_videos/test_scene.mp4")
