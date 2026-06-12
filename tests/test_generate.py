from unittest.mock import patch
from PIL import Image
from generate import load_paths_from_folder, main

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
    """Golden test: fixed seed + mock overlay → assert exact label file content."""
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"

    # Create fake background and target images
    bg_dir = tmp_path / "bgs"
    tgt_dir = tmp_path / "targets"
    bg_dir.mkdir()
    tgt_dir.mkdir()
    Image.new("RGB", (640, 480), color="blue").save(bg_dir / "bg.jpg")
    Image.new("RGBA", (50, 50), color="red").save(tgt_dir / "target.png")

    fake_labels = {"bbox": (0.5, 0.4, 0.1, 0.2)}
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
    assert len(parts) == 5  # class x y w h
    assert parts[0] == "1"  # class_id = target_idx + 1
    x, y, w, h = map(float, parts[1:])
    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0
    assert 0.0 < w <= 1.0
    assert 0.0 < h <= 1.0


def test_empty_label_on_no_placement(tmp_path):
    """If paste_single_overlay returns no bbox, label file should be empty."""
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    bg_dir = tmp_path / "bgs"
    tgt_dir = tmp_path / "targets"
    bg_dir.mkdir()
    tgt_dir.mkdir()
    Image.new("RGB", (640, 480)).save(bg_dir / "bg.jpg")
    Image.new("RGBA", (50, 50)).save(tgt_dir / "target.png")

    with patch(
        "generate.paste_single_overlay", return_value=(Image.new("RGB", (640, 480)), {})
    ):
        main(
            str(bg_dir), str(tgt_dir), None, str(img_dir), str(lbl_dir), max_attempts=5
        )

    label_file = lbl_dir / "img_00000.txt"
    assert label_file.exists()
    assert label_file.read_text() == ""  # empty label = valid YOLO negative
