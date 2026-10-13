"""Label consistency check tests."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from huanpan_perception.labelcheck import check_dataset, format_report, load_boxes


def write_image(path, width=200, height=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((height, width, 3), 120, dtype=np.uint8)).save(path)


def write_label(path, w=0.5, h=0.3, cx=0.5, cy=0.5, count=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"0 {cx} {cy} {w} {h}" for _ in range(count)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_dataset(tmp_path, sizes):
    """sizes: list of (w, h) normalised box sizes, one image each."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    for index, (w, h) in enumerate(sizes):
        name = f"img_{index:03d}"
        write_image(images / f"{name}.png")
        write_label(labels / f"{name}.txt", w=w, h=h)
    return images, labels


def test_consistent_dataset_is_not_flagged(tmp_path):
    images, labels = make_dataset(tmp_path, [(0.5, 0.3)] * 6)
    report = check_dataset(images, labels)
    assert report.images == 6
    assert report.labelled == 6
    assert report.flagged == []
    assert report.unlabelled == []
    assert report.empty_labels == []
    assert report.median_aspect == pytest.approx(0.5 * 200 / (0.3 * 100))


def test_aspect_outlier_is_flagged(tmp_path):
    images, labels = make_dataset(tmp_path, [(0.5, 0.3)] * 5 + [(0.15, 0.6)])
    report = check_dataset(images, labels)
    flagged = [item.image.name for item in report.flagged]
    assert "img_005.png" in flagged
    issue_text = " ".join(
        issue for item in report.flagged for issue in item.issues
    )
    assert "宽高比" in issue_text


def test_area_outlier_is_flagged(tmp_path):
    images, labels = make_dataset(tmp_path, [(0.5, 0.3)] * 5 + [(0.5, 0.3)])
    # Shrink the last box to a fraction of the median area.
    (labels / "img_005.txt").write_text("0 0.5 0.5 0.05 0.03\n", encoding="utf-8")
    report = check_dataset(images, labels)
    assert any(item.image.name == "img_005.png" for item in report.flagged)


def test_missing_label_is_reported(tmp_path):
    images, labels = make_dataset(tmp_path, [(0.5, 0.3)] * 3)
    (labels / "img_002.txt").unlink()
    report = check_dataset(images, labels)
    assert [p.name for p in report.unlabelled] == ["img_002.png"]
    assert any(item.image.name == "img_002.png" for item in report.flagged)


def test_empty_label_is_reported_separately_from_missing(tmp_path):
    images, labels = make_dataset(tmp_path, [(0.5, 0.3)] * 3)
    (labels / "img_001.txt").write_text("", encoding="utf-8")
    report = check_dataset(images, labels)
    assert [p.name for p in report.empty_labels] == ["img_001.png"]
    assert report.unlabelled == []


def test_odd_box_count_is_flagged(tmp_path):
    images, labels = make_dataset(tmp_path, [(0.5, 0.3)] * 5)
    (labels / "img_002.txt").write_text(
        "0 0.3 0.3 0.5 0.3\n0 0.7 0.7 0.5 0.3\n", encoding="utf-8"
    )
    report = check_dataset(images, labels)
    item = next(i for i in report.flagged if i.image.name == "img_002.png")
    assert item.box_count == 2
    assert any("框数量" in issue for issue in item.issues)


def test_tiny_box_is_flagged(tmp_path):
    images, labels = make_dataset(tmp_path, [(0.5, 0.3)] * 3)
    (labels / "img_000.txt").write_text("0 0.5 0.5 0.01 0.01\n", encoding="utf-8")
    report = check_dataset(images, labels)
    item = next(i for i in report.flagged if i.image.name == "img_000.png")
    assert any("过小" in issue for issue in item.issues)


def test_flagged_items_are_sorted_by_severity(tmp_path):
    images, labels = make_dataset(tmp_path, [(0.5, 0.3)] * 4)
    (labels / "img_000.txt").unlink()  # severity 10
    (labels / "img_001.txt").write_text("0 0.5 0.5 0.12 0.6\n", encoding="utf-8")
    report = check_dataset(images, labels)
    severities = [item.severity for item in report.flagged]
    assert severities == sorted(severities, reverse=True)
    assert report.flagged[0].image.name == "img_000.png"


def test_load_boxes_ignores_malformed_lines(tmp_path):
    images, labels = make_dataset(tmp_path, [(0.5, 0.3)])
    (labels / "img_000.txt").write_text(
        "0 0.5 0.5 0.2 0.2\nnonsense\n0 0.1 0.1 0.1\n", encoding="utf-8"
    )
    boxes = load_boxes(labels, images / "img_000.png")
    assert len(boxes) == 1
    assert boxes[0].pixel_width == pytest.approx(0.2 * 200)


def test_load_boxes_for_missing_label_is_empty(tmp_path):
    images, labels = make_dataset(tmp_path, [(0.5, 0.3)])
    (labels / "img_000.txt").unlink()
    assert load_boxes(labels, images / "img_000.png") == []


def test_format_report_mentions_the_essentials(tmp_path):
    images, labels = make_dataset(tmp_path, [(0.5, 0.3)] * 3)
    (labels / "img_001.txt").unlink()
    text = format_report(check_dataset(images, labels))
    assert "图片总数" in text
    assert "未标注" in text
    assert "img_001.png" in text


def test_load_depth_reads_16bit_millimetres_as_metres(tmp_path):
    from huanpan_perception.analyze_depth import load_depth

    path = tmp_path / "d.png"
    Image.fromarray(np.array([[450, 1000, 0]], dtype=np.uint16)).save(path)
    depth = load_depth(path)
    assert depth.dtype == np.float32
    assert depth[0, 0] == pytest.approx(0.45)
    assert depth[0, 1] == pytest.approx(1.0)
    assert depth[0, 2] == pytest.approx(0.0)


def test_depth_colour_marks_holes_in_red():
    from huanpan_perception.analyze_depth import depth_colour

    depth = np.array([[0.5, 0.0]], dtype=np.float32)
    valid = np.array([[True, False]])
    rgb = depth_colour(depth, valid)
    assert tuple(rgb[0, 1]) == (255, 0, 0)


def test_label_boxes_are_clamped_to_the_image(tmp_path):
    from huanpan_perception.analyze_depth import label_boxes

    label = tmp_path / "a.txt"
    label.write_text("0 0.5 0.5 3.0 3.0\n", encoding="utf-8")
    boxes = label_boxes(label, 100, 50)
    assert boxes == [(0, 0, 100, 50)]


def test_label_boxes_for_missing_file_is_empty(tmp_path):
    from huanpan_perception.analyze_depth import label_boxes

    assert label_boxes(tmp_path / "none.txt", 10, 10) == []
