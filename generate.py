import os
import argparse
import json
import math
from PIL import Image, ImageFilter
import random
from compositor.overlay.overlay_gate import paste_single_overlay
from compositor.overlay.overlay_multiple import paste_multiple_overlay


JERLOV_COLORS = (
    (0, 142, 207),  # I
    (0, 137, 194),  # IA
    (0, 119, 170),  # IB
    (0, 105, 139),  # II
    (0, 112, 111),  # III
    (0, 94, 39),  # IC
)


def apply_jerlov_color_augmentation(image):
    position = random.uniform(0, len(JERLOV_COLORS) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(JERLOV_COLORS) - 1)
    fraction = position - lower_index
    color = tuple(
        round(
            JERLOV_COLORS[lower_index][channel] * (1 - fraction)
            + JERLOV_COLORS[upper_index][channel] * fraction
        )
        for channel in range(3)
    )
    strength = random.uniform(0.35, 0.65)
    tint = Image.new("RGB", image.size, color)
    return Image.blend(image.convert("RGB"), tint, strength)


# LazyLoading: Loads paths instead of all the images at once so that the system doesn't crash
# useful because strings hold less memory than
def load_paths_from_folder(folder):
    paths = []
    for f in os.listdir(folder):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            paths.append(
                os.path.join(folder, f)
            )  # attached any file path that is an image within the folder.
    return sorted(paths)


def load_images_from_paths(paths):
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.copy())
    return images


def load_part_boxes(target_path):
    annotation_path = os.path.splitext(target_path)[0] + ".json"
    with open(annotation_path) as annotation_file:
        annotations = json.load(annotation_file)

    names = ("left_post", "right_post", "crossbar")
    part_boxes = {}
    for name in names:
        values = annotations.get(name)
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError(
                f"{annotation_path} must contain {name} as [x, y, w, h]"
            )
        x, y, width, height = map(float, values)
        if (
            not all(math.isfinite(value) for value in (x, y, width, height))
            or width <= 0
            or height <= 0
            or x - width / 2 < 0
            or y - height / 2 < 0
            or x + width / 2 > 1
            or y + height / 2 > 1
        ):
            raise ValueError(
                f"{annotation_path} has an invalid normalized box for {name}"
            )
        part_boxes[name] = (x, y, width, height)
    return part_boxes


def main(
    backgrounds_dir,
    real_targets_dir,
    fake_targets_dir,
    output_img_dir,
    output_yolo_dir,
    max_attempts,
    num_backgrounds=None,
    multiple=False,
):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_yolo_dir, exist_ok=True)

    background_paths = load_paths_from_folder(backgrounds_dir)
    real_target_paths = sorted(load_paths_from_folder(real_targets_dir))
    fake_target_paths = load_paths_from_folder(fake_targets_dir) if fake_targets_dir else []

    if fake_target_paths and not multiple:
        raise ValueError("fake_targets_dir can only be used with multiple=True")

    if multiple:
        real_targets = load_images_from_paths(real_target_paths)
        fake_targets = load_images_from_paths(fake_target_paths)

    if num_backgrounds is not None:
        if num_backgrounds < 0:
            raise ValueError("num_backgrounds must be non-negative")
        if num_backgrounds and not background_paths:
            raise ValueError("Cannot generate images: backgrounds directory is empty")
        background_paths = [
            background_paths[index % len(background_paths)]
            for index in range(num_backgrounds)
        ]

    for i, bg_path in enumerate(background_paths):
        if multiple:
            with Image.open(bg_path) as bg:
                composite, labels = paste_multiple_overlay(
                    bg.copy(),
                    real_targets,
                    fake_targets,
                    max_attempts=max_attempts,
                )
        else:
            target_idx = i % len(real_target_paths)
            class_id = target_idx + 1
            target_path = real_target_paths[target_idx]
            part_boxes = load_part_boxes(target_path)
            with Image.open(bg_path) as bg:
                with Image.open(target_path) as target:
                    composite, labels = paste_single_overlay(
                        bg.copy(),
                        target.copy(),
                        class_id=class_id,
                        part_boxes=part_boxes,
                        max_attempts=max_attempts,
                    )

        img_name = f"img_{i:05d}.jpg"
        label_name = f"img_{i:05d}.txt"

        img_path = os.path.join(output_img_dir, img_name)
        yolo_path = os.path.join(output_yolo_dir, label_name)

        if composite is None or labels is None:
            print(f"Warning: placement failed for {img_name}, saving empty label")
            open(yolo_path, "w").close()
            continue

        composite = apply_jerlov_color_augmentation(composite)
        blur_radius = random.uniform(0.6, 1.2)
        composite = composite.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        composite.save(img_path, quality=95)

        with open(yolo_path, "w") as f:
            if multiple:
                for label in labels:
                    label_class_id = label["class_id"]
                    x, y, w, h = label["bbox"]
                    f.write(
                        f"{label_class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                    )
            else:
                bbox = labels.get("bbox")
                keypoints = labels.get("keypoints")
                if not bbox or not keypoints:
                    print(f"Warning: no pose placement for {img_name}, saving empty label")
                    continue
                x, y, w, h = bbox
                pose_values = " ".join(
                    f"{coord:.6f}" for point in keypoints for coord in point
                )
                part_values = " ".join(
                    f"{coord:.6f}"
                    for name in ("left_post", "right_post", "crossbar")
                    for coord in labels["part_boxes"][name]
                )
                f.write(
                    f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} "
                    f"{pose_values} {part_values}\n"
                )

        print(f"Generated {img_name} + {label_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--backgrounds_dir", required=True)
    parser.add_argument("--real_targets_dir", required=True)
    parser.add_argument("--fake_targets_dir", default=None)
    parser.add_argument(
        "--output_img_dir",
        default="output_images",
        help="Output folder for generated images",
    )
    parser.add_argument(
        "--output_yolo_dir",
        default="output_yolo",
        help="Output folder for YOLO labels",
    )
    parser.add_argument("--max_attempts", type=int, default=20)
    parser.add_argument("--num_backgrounds", type=int, default=None)
    parser.add_argument(
        "--multiple",
        action="store_true",
        help="Overlay all real targets instead of cycling one target per image",
    )

    args = parser.parse_args()

    main(
        args.backgrounds_dir,
        args.real_targets_dir,
        args.fake_targets_dir,
        args.output_img_dir,
        args.output_yolo_dir,
        args.max_attempts,
        args.num_backgrounds,
        args.multiple,
    )
