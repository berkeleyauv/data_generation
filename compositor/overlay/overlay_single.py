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


def paste_single_overlay(
    background: Image.Image,
    overlay: Image.Image,
    class_id: int,
    scale_range=(0.2, 0.6),
    max_attempts=20,
    color_match_strength=0.05,
):
    """
    Paste a single target image onto a background with:
        - slight perspective warp
        - subtle color matching
        - max overlay area <= 40% of background
    """
    bg_w, bg_h = background.size
    bg_area = bg_w * bg_h

    overlay = overlay.convert("RGBA")
    bg = background.convert("RGBA")

    for attempt in range(max_attempts):
        scale = random.uniform(*scale_range)
        new_w = int(bg_w * scale)
        new_h = int(new_w * overlay.height / overlay.width)

        if new_w * new_h > 0.4 * bg_area:
            continue

        ov = overlay.resize((new_w, new_h), Image.Resampling.LANCZOS)
        ov_np = np.array(ov)
        ov_np = random_perspective_transform(ov_np, max_offset=0.1)  # warp back to 0.2
        ov = Image.fromarray(ov_np)

        if ov.width >= bg_w or ov.height >= bg_h:
            continue

        x_min = random.randint(0, bg_w - ov.width)
        y_min = random.randint(0, bg_h - ov.height)

        # --- Subtle color matching ---
        bg_crop = bg.crop((x_min, y_min, x_min + ov.width, y_min + ov.height)).convert(
            "RGB"
        )
        ov_rgb = ov.convert("RGB")

        bg_np = np.array(bg_crop).astype(np.float32)
        ov_np_rgb = np.array(ov_rgb).astype(np.float32)

        ov_mean = ov_np_rgb.mean(axis=(0, 1))
        bg_mean = bg_np.mean(axis=(0, 1))

        # Apply small shift toward background color
        ov_np_rgb = ov_np_rgb + (bg_mean - ov_mean) * color_match_strength
        ov_np_rgb = np.clip(ov_np_rgb, 0, 255).astype(np.uint8)
        ov_rgb = Image.fromarray(ov_np_rgb)

        # Paste overlay
        bg.paste(ov_rgb, (x_min, y_min), ov)

        x_center = (x_min + ov.width / 2) / bg_w
        y_center = (y_min + ov.height / 2) / bg_h
        w_norm = ov.width / bg_w
        h_norm = ov.height / bg_h

        label = {"class_id": class_id, "bbox": (x_center, y_center, w_norm, h_norm)}
        return bg.convert("RGB"), label

    return None, None
