# gate_overlay

Generate synthetic gate images by compositing cropped gate targets onto background images. This branch supports the gate-overlay workflow only.

## Supported Workflow

`generate.py` uses `compositor/overlay/overlay_gate.py` to resize, perspective-warp, color-match, and blur each gate composite. The gate's average color is strongly shifted toward its local background, and a randomized Jerlov water tint is applied to the full composite. The tint is sampled across water types I through IC, with a blend strength of 35% to 65%.

The legacy torpedo overlay, old single-overlay path, and multiple-overlay path are outdated and unsupported on this branch:

- `compositor/overlay/overlay_torpedo.py`
- `compositor/overlay/overlay_single.py`
- `compositor/overlay/overlay_multiple.py`

Use the gate-overlay workflow below; do not use the legacy torpedo, single, or multiple workflows.

## Inputs

Place background images in `backgrounds/` and cropped gate images in `targets_original/`. Every gate image needs a same-stem JSON sidecar containing normalized component boxes. For example, `gate_01.png` needs `gate_01.json`:

```json
{
  "left_post": [0.05, 0.5, 0.08, 0.9],
  "right_post": [0.95, 0.5, 0.08, 0.9],
  "crossbar": [0.5, 0.08, 0.9, 0.08]
}
```

Each box is `[center_x, center_y, width, height]`, normalized to the source gate image. Values must be between 0 and 1, and width and height must be positive.

## Generate Images

Run from the repository root. By default, images and labels are written to `output_images/` and `output_yolo/`. Repeated runs reuse these same folders and overwrite files with matching names; no new per-run folders are created.

```bash
python generate.py \
  --backgrounds_dir backgrounds \
  --real_targets_dir targets_original \
  --max_attempts 20 \
  --num_backgrounds 10
```

Use `--output_img_dir` or `--output_yolo_dir` only when intentionally choosing different output locations. Without `--num_backgrounds`, the pipeline creates one output for each available background. When the requested count is larger than the number of available backgrounds, it processes backgrounds in filename order and cycles through them round-robin. Target class IDs are assigned starting at 1 in sorted target-filename order.

## Label Format

Each label row contains 29 values in this order:

```text
class x y w h
x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
x_l y_l w_l h_l x_r y_r w_r h_r x_c y_c w_c h_c
```

All coordinates and box sizes are normalized to the generated image. The gate box is `[center_x, center_y, width, height]`. The four keypoints are the warped gate corners in top-left, top-right, bottom-right, bottom-left order; visibility is `2` when visible and `0` when outside the image. The final three boxes are left post, right post, and crossbar, each in `[center_x, center_y, width, height]` order.

This 29-value row is project-specific, not a standard YOLO pose row. Training code must use a parser that understands the appended component boxes.

## Tests

Run the test suite with:

```bash
pytest tests/ -v
```