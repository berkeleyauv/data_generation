import json
from unittest.mock import patch
import numpy as np
from PIL import Image
from generate import load_paths_from_folder, main
from compositor.overlay.overlay_gate import paste_single_overlay


def write_part_annotations(target_path):
    target_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "left_post": [0.25, 0.5, 0.1, 0.8],
                "right_post": [0.75, 0.5, 0.1, 0.8],
                "crossbar": [0.5, 0.15, 0.8, 0.1],
            }
        )
    )

# pytest tests/ -v

# --- load_paths_from_folder ---


def test_load_paths_finds_images(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "b.PNG").write_bytes(b"")
    (tmp_path / "ignore.txt").write_bytes(b"")
    paths = load_paths_from_folder(str(tmp_path))
    assert len(paths) == 2
    assert all(p.endswith((".jpg", ".PNG")) for p in paths)


def test_load_paths_empty_folder(tmp_path):
    assert load_paths_from_folder(str(tmp_path)) == []


# --- YOLO label format ---


def test_yolo_label_written_correctly(tmp_path):
    """Golden test: mock overlay → assert exact label file content."""
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"

    # Create fake background and target images
    bg_dir = tmp_path / "bgs"
    tgt_dir = tmp_path / "targets"
    bg_dir.mkdir()
    tgt_dir.mkdir()
    Image.new("RGB", (640, 480), color="blue").save(bg_dir / "bg.jpg")
    target_path = tgt_dir / "target.png"
    Image.new("RGBA", (50, 50), color="red").save(target_path)
    write_part_annotations(target_path)

    fake_labels = {
        "bbox": (0.5, 0.4, 0.1, 0.2),
        "keypoints": (
            (0.45, 0.3, 2),
            (0.55, 0.3, 2),
            (0.55, 0.5, 2),
            (0.45, 0.5, 2),
        ),
        "part_boxes": {
            "left_post": (0.25, 0.5, 0.1, 0.8),
            "right_post": (0.75, 0.5, 0.1, 0.8),
            "crossbar": (0.5, 0.15, 0.8, 0.1),
        },
    }
    fake_composite = Image.new("RGB", (640, 480))

    with patch(
        "generate.paste_single_overlay", return_value=(fake_composite, fake_labels)
    ):
        main(
            str(bg_dir), str(tgt_dir), None, str(img_dir), str(lbl_dir), max_attempts=5
        )

    label_file = lbl_dir / "img_00000.txt"
    assert label_file.exists()
    line = label_file.read_text().strip()
    parts = line.split()
    assert len(parts) == 29  # pose row plus three component boxes
    assert parts[0] == "1"  # class_id = target_idx + 1
    x, y, w, h = map(float, parts[1:5])
    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0
    assert 0.0 < w <= 1.0
    assert 0.0 < h <= 1.0
    assert list(map(float, parts[5:])) == [
        0.45, 0.3, 2.0, 0.55, 0.3, 2.0, 0.55, 0.5, 2.0, 0.45, 0.5, 2.0,
        0.25, 0.5, 0.1, 0.8, 0.75, 0.5, 0.1, 0.8, 0.5, 0.15, 0.8, 0.1,
    ]


def test_empty_label_on_no_placement(tmp_path):
    """If paste_single_overlay returns no bbox, label file should be empty."""
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    bg_dir = tmp_path / "bgs"
    tgt_dir = tmp_path / "targets"
    bg_dir.mkdir()
    tgt_dir.mkdir()
    Image.new("RGB", (640, 480)).save(bg_dir / "bg.jpg")
    target_path = tgt_dir / "target.png"
    Image.new("RGBA", (50, 50)).save(target_path)
    write_part_annotations(target_path)

    with patch(
        "generate.paste_single_overlay", return_value=(Image.new("RGB", (640, 480)), {})
    ):
        main(
            str(bg_dir), str(tgt_dir), None, str(img_dir), str(lbl_dir), max_attempts=5
        )

    label_file = lbl_dir / "img_00000.txt"
    assert label_file.exists()
    assert label_file.read_text() == ""  # empty label = valid YOLO negative


def test_reuses_backgrounds_to_match_requested_count(tmp_path):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    bg_dir = tmp_path / "bgs"
    tgt_dir = tmp_path / "targets"
    bg_dir.mkdir()
    tgt_dir.mkdir()
    Image.new("RGB", (640, 480), color="blue").save(bg_dir / "bg.jpg")
    target_path = tgt_dir / "target.png"
    Image.new("RGBA", (50, 50), color="red").save(target_path)
    write_part_annotations(target_path)

    fake_labels = {
        "bbox": (0.5, 0.4, 0.1, 0.2),
        "keypoints": ((0.45, 0.3, 2), (0.55, 0.3, 2), (0.55, 0.5, 2), (0.45, 0.5, 2)),
        "part_boxes": {
            "left_post": (0.25, 0.5, 0.1, 0.8),
            "right_post": (0.75, 0.5, 0.1, 0.8),
            "crossbar": (0.5, 0.15, 0.8, 0.1),
        },
    }
    fake_composite = Image.new("RGB", (640, 480))

    with patch(
        "generate.paste_single_overlay", return_value=(fake_composite, fake_labels)
    ) as paste_overlay:
        main(
            str(bg_dir),
            str(tgt_dir),
            None,
            str(img_dir),
            str(lbl_dir),
            max_attempts=5,
            num_backgrounds=3,
        )

    assert paste_overlay.call_count == 3
    assert len(list(img_dir.glob("*.jpg"))) == 3
    assert len(list(lbl_dir.glob("*.txt"))) == 3


def test_single_mode_cycles_through_targets(tmp_path):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    bg_dir = tmp_path / "bgs"
    tgt_dir = tmp_path / "targets"
    bg_dir.mkdir()
    tgt_dir.mkdir()
    Image.new("RGB", (640, 480), color="blue").save(bg_dir / "bg.jpg")
    target_a = tgt_dir / "a.png"
    target_b = tgt_dir / "b.png"
    Image.new("RGBA", (50, 50), color=(255, 0, 0, 255)).save(target_a)
    Image.new("RGBA", (50, 50), color=(0, 255, 0, 255)).save(target_b)
    write_part_annotations(target_a)
    write_part_annotations(target_b)

    fake_labels = {
        "bbox": (0.5, 0.4, 0.1, 0.2),
        "keypoints": ((0.45, 0.3, 2), (0.55, 0.3, 2), (0.55, 0.5, 2), (0.45, 0.5, 2)),
        "part_boxes": {
            "left_post": (0.25, 0.5, 0.1, 0.8),
            "right_post": (0.75, 0.5, 0.1, 0.8),
            "crossbar": (0.5, 0.15, 0.8, 0.1),
        },
    }
    fake_composite = Image.new("RGB", (640, 480))
    with patch(
        "generate.paste_single_overlay",
        return_value=(fake_composite, fake_labels),
    ) as paste_overlay:
        main(
            str(bg_dir),
            str(tgt_dir),
            None,
            str(img_dir),
            str(lbl_dir),
            max_attempts=5,
            num_backgrounds=4,
        )

    target_colors = [
        call.args[1].getpixel((0, 0)) for call in paste_overlay.call_args_list
    ]
    assert target_colors == [
        (255, 0, 0, 255),
        (0, 255, 0, 255),
        (255, 0, 0, 255),
        (0, 255, 0, 255),
    ]


def test_multiple_mode_writes_detection_labels(tmp_path):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    bg_dir = tmp_path / "bgs"
    tgt_dir = tmp_path / "targets"
    bg_dir.mkdir()
    tgt_dir.mkdir()
    Image.new("RGB", (640, 480), color="blue").save(bg_dir / "bg.jpg")
    Image.new("RGBA", (50, 50), color="red").save(tgt_dir / "target.png")

    fake_composite = Image.new("RGB", (640, 480))
    fake_labels = [{"class_id": 1, "bbox": (0.5, 0.4, 0.1, 0.2)}]
    with patch(
        "generate.paste_multiple_overlay",
        return_value=(fake_composite, fake_labels),
    ) as paste_overlay:
        main(
            str(bg_dir),
            str(tgt_dir),
            None,
            str(img_dir),
            str(lbl_dir),
            max_attempts=5,
            multiple=True,
        )

    assert paste_overlay.call_count == 1
    assert (lbl_dir / "img_00000.txt").read_text().split() == [
        "1",
        "0.500000",
        "0.400000",
        "0.100000",
        "0.200000",
    ]


def test_oversized_gate_is_cropped_and_pose_labels_stay_in_frame():
    background = Image.new("RGB", (100, 100), color="blue")
    target = Image.new("RGBA", (10, 10), color="red")

    def identity_transform(image, max_offset=0.3, return_corners=False):
        height, width = image.shape[:2]
        corners = np.float32(
            [[0, 0], [width, 0], [width, height], [0, height]]
        )
        return (image, corners) if return_corners else image

    with patch(
        "compositor.overlay.overlay_gate.random.uniform", return_value=1.2
    ), patch(
        "compositor.overlay.overlay_gate.random.randint", return_value=-10
    ), patch(
        "compositor.overlay.overlay_gate.random_perspective_transform",
        side_effect=identity_transform,
    ):
        composite, label = paste_single_overlay(
            background,
            target,
            class_id=1,
            part_boxes={"left_post": (0.25, 0.5, 0.1, 0.8)},
        )

    assert composite.size == background.size
    assert label["bbox"] == (0.5, 0.5, 1.0, 1.0)
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y, _ in label["keypoints"])
    assert all(visibility == 0.0 for _, _, visibility in label["keypoints"])
    np.testing.assert_allclose(
        label["part_boxes"]["left_post"], (0.2, 0.5, 0.12, 0.96), atol=1e-6
    )
