import os
import argparse
from PIL import Image
from datetime import datetime
import importlib
from compositor.augment import augment


def load_overlay(overlay_str):
    module_name, func_name = overlay_str.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def save_label(label_dict, label_path):
    with open(label_path, "w") as f:
        if isinstance(label_dict, list):
            for lbl in label_dict:
                class_id = lbl['class_id']
                xc, yc, w, h = lbl['bbox']
                f.write(f"{class_id} {xc} {yc} {w} {h}\n")
        else:
            class_id = label_dict['class_id']
            xc, yc, w, h = label_dict['bbox']
            f.write(f"{class_id} {xc} {yc} {w} {h}\n")


def main(backgrounds_dir, targets_dir, output_dir, overlay_func_str, max_attempts):
    if not os.path.isdir(targets_dir):
        raise ValueError(f"Targets dir {targets_dir} is not a folder")
    target_files = [f for f in os.listdir(targets_dir)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if not target_files:
        raise ValueError("No valid image files found in targets_dir")

    target_paths = {os.path.splitext(f)[0]: os.path.join(targets_dir, f)
                    for f in sorted(target_files)}
    class_map = {name: idx for idx, name in enumerate(target_paths.keys())}

    bg_files = [os.path.join(backgrounds_dir, f) for f in os.listdir(backgrounds_dir)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if not bg_files:
        raise ValueError("No valid background images found")

    overlay_fn = load_overlay(overlay_func_str)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_images_dir = os.path.join(output_dir, f"{timestamp}/images")
    output_labels_dir = os.path.join(output_dir, f"{timestamp}/labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    target_names = list(target_paths.keys())
    i = 0

    for bg_idx, bg_path in enumerate(sorted(bg_files)):
        bg = Image.open(bg_path).convert("RGB")

        # Determine if overlay expects multiple or single
        if 'multiple' in overlay_func_str.lower():
            overlays = [Image.open(target_paths[name]).convert("RGBA") for name in target_names]
            composite, labels = overlay_fn(bg, overlays, max_attempts=max_attempts)
            if composite is None:
                print(f"Skipping background {bg_idx} because overlays couldn't be placed")
                continue
            image_filename = f"sample_{i:05d}.jpg"
            label_filename = f"sample_{i:05d}.txt"
            composite.save(os.path.join(output_images_dir, image_filename))
            save_label(labels, os.path.join(output_labels_dir, label_filename))
            print(f"[{i}] Saved {image_filename} with {len(labels)} targets")
        else:
            # Single overlay alternating targets
            class_name = target_names[i % len(target_names)]
            class_id = class_map[class_name]
            target_img = Image.open(target_paths[class_name]).convert("RGBA")
            target_img_aug = augment(target_img)
            composite, label = overlay_fn(bg, target_img_aug, class_id, max_attempts=max_attempts)
            if composite is None:
                print(f"Skipping background {bg_idx} because overlay couldn't be placed")
                continue
            image_filename = f"sample_{i:05d}.jpg"
            label_filename = f"sample_{i:05d}.txt"
            composite.save(os.path.join(output_images_dir, image_filename))
            save_label(label, os.path.join(output_labels_dir, label_filename))
            print(f"[{i}] Saved {image_filename} with 1 target: {class_name}")

        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmented overlay dataset")
    parser.add_argument("--backgrounds_dir", type=str, required=True)
    parser.add_argument("--targets_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--overlay_func", type=str, required=True)
    parser.add_argument("--max_attempts", type=int, default=20)
    args = parser.parse_args()

    main(args.backgrounds_dir, args.targets_dir, args.output_dir, args.overlay_func, args.max_attempts)
