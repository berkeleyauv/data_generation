# overlay_multiple.py
import random
import numpy as np
from PIL import Image
import cv2


def random_perspective_transform(img_np, max_offset=0.1):
    h, w = img_np.shape[:2]
    dx = int(w * max_offset)
    dy = int(h * max_offset)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32(
        [
            [random.randint(0, dx), random.randint(0, dy)],
            [w - random.randint(0, dx), random.randint(0, dy)],
            [w - random.randint(0, dx), h - random.randint(0, dy)],
            [random.randint(0, dx), h - random.randint(0, dy)],
        ]
    )

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        img_np, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )
    return warped


def match_hue_subtle(overlay_np, background_np):
    if overlay_np.shape[2] == 4:
        alpha = overlay_np[..., 3].copy()
    else:
        alpha = None

    ov_rgb = overlay_np[..., :3].astype(np.float32)
    bg_rgb = background_np[..., :3].astype(np.float32)

    ov_mean = ov_rgb.mean(axis=(0, 1))
    bg_mean = bg_rgb.mean(axis=(0, 1))

    ov_rgb += 0.05 * (bg_mean - ov_mean)
    ov_rgb = np.clip(ov_rgb, 0, 255).astype(np.uint8)

    if alpha is not None:
        overlay_np = np.dstack([ov_rgb, alpha])
    else:
        overlay_np = ov_rgb

    return overlay_np


def check_overlap(new_box, existing_boxes):
    x1, y1, w1, h1 = new_box
    for x2, y2, w2, h2 in existing_boxes:
        if not (x1 + w1 <= x2 or x1 >= x2 + w2 or y1 + h1 <= y2 or y1 >= y2 + h2):
            return True
    return False


def paste_multiple_overlay(background, real_overlays, fake_overlays, max_attempts=30):
    bg = background.convert("RGBA")
    bg_np = np.array(bg)
    placed_boxes = []
    labels = []

    # ✦ Randomly select ONE fake overlay
    fake_choice = []
    if fake_overlays:
        fake_choice = [random.choice(fake_overlays)]

    # ✦ Assign class IDs
    combined = []

    # real overlays get class IDs starting from 1
    for idx, ov in enumerate(real_overlays):
        combined.append((idx + 1, ov))  # 1, 2, 3, ...

    # fake overlays get class ID 0
    for ov in fake_choice:
        combined.append((0, ov))

    for class_id, ov_img in combined:
        ov = ov_img.convert("RGBA")
        success = False

        for attempt in range(max_attempts):
            scale = random.uniform(0.15, 0.35)
            new_w = int(bg.width * scale)
            aspect = ov.width / ov.height
            new_h = int(new_w / aspect)

            ov_resized = ov.resize((new_w, new_h), Image.Resampling.LANCZOS)

            angle = random.uniform(-15, 15)
            ov_resized = ov_resized.rotate(angle, expand=True)

            ov_np = np.array(ov_resized)
            ov_np = random_perspective_transform(ov_np, max_offset=0.08)
            ov_np = match_hue_subtle(ov_np, bg_np)

            ov_resized = Image.fromarray(ov_np)
            new_w, new_h = ov_resized.size

            if new_w >= bg.width or new_h >= bg.height:
                continue

            x_min = random.randint(0, bg.width - new_w)
            y_min = random.randint(0, bg.height - new_h)
            new_box = (x_min, y_min, new_w, new_h)

            if not check_overlap(new_box, placed_boxes):
                alpha = (
                    ov_np[..., 3] / 255.0
                    if ov_np.shape[2] == 4
                    else np.ones((new_h, new_w))
                )

                for c in range(3):
                    bg_np[y_min : y_min + new_h, x_min : x_min + new_w, c] = (
                        alpha * ov_np[..., c]
                        + (1 - alpha)
                        * bg_np[y_min : y_min + new_h, x_min : x_min + new_w, c]
                    )

                placed_boxes.append(new_box)

                x_center = (x_min + new_w / 2) / bg.width
                y_center = (y_min + new_h / 2) / bg.height
                w_norm = new_w / bg.width
                h_norm = new_h / bg.height

                labels.append(
                    {"class_id": class_id, "bbox": (x_center, y_center, w_norm, h_norm)}
                )

                success = True
                break

        if not success:
            print(f"Warning: Could not place overlay after {max_attempts} attempts.")

    composite = Image.fromarray(bg_np).convert("RGB")
    return composite, labels
