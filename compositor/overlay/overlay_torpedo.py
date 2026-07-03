import os
from PIL import Image, ImageDraw


def draw_ring(img, center, radius=30, color=(255, 0, 0), border_ratio=0.25):
    x, y = center
    w_img, h_img = img.size

    width_px = max(1, int(radius * border_ratio))

    x = max(radius, min(w_img - radius, x))
    y = max(radius, min(h_img - radius, y))

    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw = ImageDraw.Draw(img)

    draw.ellipse(bbox, fill=(0, 0, 0))
    draw.ellipse(bbox, outline=color, width=width_px)

    box_w = 2 * radius
    box_h = 2 * radius
    return x, y, box_w, box_h


def main(images_dir, yolo_dir, output_dir, output_yolo_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_yolo_dir, exist_ok=True)

    for label_file in os.listdir(yolo_dir):
        if not label_file.endswith(".txt"):
            continue

        img_name = label_file.replace(".txt", ".jpg")
        img_path = os.path.join(images_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Image not found for label {label_file}")
            continue

        img = Image.open(img_path)
        w_img, h_img = img.size

        # Read all target boxes (class 1 or 2)
        targets = []
        with open(os.path.join(yolo_dir, label_file)) as f:
            for line in f:
                cls, x, y, w, h = line.strip().split()
                cls = int(cls)
                if cls not in (1, 2):
                    continue
                x, y, w, h = map(float, (x, y, w, h))
                targets.append((cls, x, y, w, h))

        yolo_out_path = os.path.join(output_yolo_dir, label_file)
        out_path = os.path.join(output_dir, img_name)

        out_lines = []

        for cls, x, y, w, h in targets:
            box_w_px = w * w_img
            box_h_px = h * h_img
            box_x_px = x * w_img
            box_y_px = y * h_img

            left = box_x_px - box_w_px / 2
            top = box_y_px - box_h_px / 2
            right = box_x_px + box_w_px / 2

            ring_center_x = (left + right) / 2
            ring_center_y = top + box_h_px * 0.75

            ring_x, ring_y, ring_w, ring_h = draw_ring(
                img, (ring_center_x, ring_center_y)
            )

            x_norm = ring_x / w_img
            y_norm = ring_y / h_img
            w_norm = ring_w / w_img
            h_norm = ring_h / h_img

            # write ring bbox with same class as the target
            out_lines.append(
                f"{cls} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

        with open(yolo_out_path, "w") as f:
            f.write("\n".join(out_lines) + "\n")

        img.save(out_path)
        print(f"Processed {img_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--yolo_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_yolo_dir", required=True)

    args = parser.parse_args()
    main(args.images_dir, args.yolo_dir, args.output_dir, args.output_yolo_dir)
