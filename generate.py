import os
import argparse
from PIL import Image
from datetime import datetime
import importlib
from compositor.augment import augment


def load_overlay(overlay_str):
    """Dynamically import the specified overlay function."""
    module_name, func_name = overlay_str.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def save_label(label_dict, label_path):
    """Write YOLO-format label file."""
    class_id = label_dict['class_id']
    xc, yc, w, h = label_dict['bbox']
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {xc} {yc} {w} {h}\n")


def main(backgrounds_dir, targets_dir, output_dir, overlay_func_str):
    # --- Validate target directory ---
    if not os.path.isdir(targets_dir):
        raise ValueError(f"Targets dir {targets_dir} is not a folder")

    target_files = [f for f in os.listdir(targets_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not target_files:
        raise ValueError("No valid image files found in targets_dir")

    # --- Build dynamic class map ---
    target_paths = {os.path.splitext(f)[0]: os.path.join(targets_dir, f)
                    for f in sorted(target_files)}
    class_map = {name: idx for idx, name in enumerate(target_paths.keys())}

    # --- Collect backgrounds ---
    bg_files = [os.path.join(backgrounds_dir, f) for f in os.listdir(backgrounds_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not bg_files:
        raise ValueError("No valid background images found")

    num_targets = len(target_paths)
    num_backgrounds = len(bg_files)
    overlay_fn = load_overlay(overlay_func_str)

    print(f"Found {num_backgrounds} backgrounds and {num_targets} targets.")
    print("Each background will be used exactly once.")
    print(f"Distributing {num_backgrounds} backgrounds across {num_targets} targets...")

    # --- Output dirs ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_images_dir = os.path.join(output_dir, f'{timestamp}/images')
    output_labels_dir = os.path.join(output_dir, f'{timestamp}/labels')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # --- Generation loop (each background used once) ---
    i = 0
    target_names = list(target_paths.keys())

    for bg_idx, bg_path in enumerate(sorted(bg_files)):
        # Pick target in round-robin fashion
        class_name = target_names[bg_idx % num_targets]
        class_id = class_map[class_name]
        target_path = target_paths[class_name]

        # Load and process images
        bg = Image.open(bg_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGBA")
        target_img_aug = augment(target_img)

        composite, label = overlay_fn(bg, target_img_aug, class_id)
        if composite is None:
            continue

        # Save output
        image_filename = f"sample_{i:05d}.jpg"
        label_filename = f"sample_{i:05d}.txt"
        composite.save(os.path.join(output_images_dir, image_filename))
        save_label(label, os.path.join(output_labels_dir, label_filename))

        print(f"[{i}] Saved {image_filename} using background {os.path.basename(bg_path)} "
              f"and target {class_name}")
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate augmented overlay dataset")
    parser.add_argument('--backgrounds_dir', type=str, required=True)
    parser.add_argument('--targets_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--overlay_func', type=str, required=True,
                        help="Overlay function path, e.g. 'compositor.overlay.overlay_gate.paste_overlay'")
    args = parser.parse_args()

    main(args.backgrounds_dir, args.targets_dir, args.output_dir, args.overlay_func)
