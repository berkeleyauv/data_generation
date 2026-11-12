import random
import numpy as np
from PIL import Image
import cv2


def random_perspective_transform(img_np, max_offset=0.2, return_corners=False):
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
    if return_corners:
        return warped, dst
    return warped


def paste_single_overlay(
    background: Image.Image,
    overlay: Image.Image,
    class_id: int,
    scale_range=(0.2, 0.9),
    max_attempts=20,
    color_match_strength=0.3,
    part_boxes=None,
):
    """
    Paste a perspective-warped target onto a background, allowing it to extend
    beyond the frame edges.
    """
    bg_w, bg_h = background.size
    overlay = overlay.convert("RGBA")
    bg = background.convert("RGBA")

    for attempt in range(max_attempts):
        scale = random.uniform(*scale_range)
        new_w = int(bg_w * scale)
        new_h = int(new_w * overlay.height / overlay.width)

        ov = overlay.resize((new_w, new_h), Image.Resampling.LANCZOS)
        ov_np = np.array(ov)
        ov_np, corners = random_perspective_transform(
            ov_np, max_offset=0.2, return_corners=True
        )
        ov = Image.fromarray(ov_np)

        x_min = random.randint(min(0, bg_w - ov.width), max(0, bg_w - ov.width))
        y_min = random.randint(min(0, bg_h - ov.height), max(0, bg_h - ov.height))

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

        corners = corners + np.array([x_min, y_min], dtype=np.float32)
        clipped_corners = np.clip(corners, [0, 0], [bg_w, bg_h])
        x_min_box, y_min_box = clipped_corners.min(axis=0)
        x_max_box, y_max_box = clipped_corners.max(axis=0)
        x_center = (x_min_box + x_max_box) / (2 * bg_w)
        y_center = (y_min_box + y_max_box) / (2 * bg_h)
        w_norm = (x_max_box - x_min_box) / bg_w
        h_norm = (y_max_box - y_min_box) / bg_h
        keypoints = tuple(
            (
                float(np.clip(x / bg_w, 0, 1)),
                float(np.clip(y / bg_h, 0, 1)),
                2.0 if 0 <= x <= bg_w and 0 <= y <= bg_h else 0.0,
            )
            for x, y in corners
        )

        transformed_part_boxes = {}
        if part_boxes is not None:
            source_corners = np.float32(
                [[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]]
            )
            placement_offset = np.array([x_min, y_min], dtype=np.float32)
            transform = cv2.getPerspectiveTransform(
                source_corners, corners - placement_offset
            )
            for name, (part_x_center, part_y_center, width, height) in part_boxes.items():
                part_corners = np.float32(
                    [[
                        (part_x_center - width / 2) * new_w,
                        (part_y_center - height / 2) * new_h,
                    ], [
                        (part_x_center + width / 2) * new_w,
                        (part_y_center - height / 2) * new_h,
                    ], [
                        (part_x_center + width / 2) * new_w,
                        (part_y_center + height / 2) * new_h,
                    ], [
                        (part_x_center - width / 2) * new_w,
                        (part_y_center + height / 2) * new_h,
                    ]]
                )
                mapped = cv2.perspectiveTransform(
                    part_corners.reshape(1, -1, 2), transform
                )[0]
                mapped += placement_offset
                x_min_part = np.clip(mapped[:, 0].min(), 0, bg_w)
                y_min_part = np.clip(mapped[:, 1].min(), 0, bg_h)
                x_max_part = np.clip(mapped[:, 0].max(), 0, bg_w)
                y_max_part = np.clip(mapped[:, 1].max(), 0, bg_h)
                transformed_part_boxes[name] = (
                    float((x_min_part + x_max_part) / (2 * bg_w)),
                    float((y_min_part + y_max_part) / (2 * bg_h)),
                    float((x_max_part - x_min_part) / bg_w),
                    float((y_max_part - y_min_part) / bg_h),
                )

        label = {
            "class_id": class_id,
            "bbox": (x_center, y_center, w_norm, h_norm),
            "keypoints": keypoints,
        }
        if part_boxes is not None:
            label["part_boxes"] = transformed_part_boxes
        return bg.convert("RGB"), label

    return None, None
