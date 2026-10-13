"""Dataset split layout tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from huanpan_perception.dataset import build_split, list_images, read_label_classes


def make_frame(path, width=32, height=24, channels=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    if channels == 3:
        array = np.full((height, width, 3), 128, dtype=np.uint8)
    else:
        array = np.full((height, width), 128, dtype=np.uint8)
    Image.fromarray(array).save(path)


def write_label(path, class_id=0, cx=0.5, cy=0.5, w=0.4, h=0.3):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{class_id} {cx} {cy} {w} {h}\n", encoding="utf-8")


def test_build_split_creates_ultralytics_layout(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    for index in range(10):
        make_frame(images / f"capture_{index:05d}.png")
        write_label(labels / f"capture_{index:05d}.txt")
    # One unlabelled frame must be skipped.
    make_frame(images / "capture_unlabelled.png")

    stats = build_split(
        images, labels, tmp_path / "data", class_names=["server_unit"], val_ratio=0.2, seed=0
    )

    assert stats.train == 8
    assert stats.val == 2
    assert stats.unlabeled == 1
    assert stats.class_counts == {0: 10}

    assert len(list_images(tmp_path / "data" / "images" / "train")) == 8
    assert len(list_images(tmp_path / "data" / "images" / "val")) == 2
    assert len(list((tmp_path / "data" / "labels" / "train").glob("*.txt"))) == 8

    data_yaml = stats.data_yaml.read_text(encoding="utf-8")
    assert "images/train" in data_yaml
    assert "images/val" in data_yaml
    assert "nc: 1" in data_yaml
    assert "0: server_unit" in data_yaml


def test_build_split_is_reproducible(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    for index in range(8):
        make_frame(images / f"f{index}.png")
        write_label(labels / f"f{index}.txt")

    first = build_split(images, labels, tmp_path / "a", class_names=["server_unit"], seed=1)
    second = build_split(images, labels, tmp_path / "b", class_names=["server_unit"], seed=1)
    assert first.train == second.train
    assert sorted(p.name for p in list_images(tmp_path / "a" / "images" / "val")) == sorted(
        p.name for p in list_images(tmp_path / "b" / "images" / "val")
    )


def test_build_split_without_labels_is_an_error(tmp_path):
    images = tmp_path / "images"
    make_frame(images / "only.png")
    with pytest.raises(FileNotFoundError):
        build_split(
            images, tmp_path / "labels", tmp_path / "data", class_names=["server_unit"]
        )


def test_build_split_with_no_images_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_split(
            tmp_path / "images",
            tmp_path / "labels",
            tmp_path / "data",
            class_names=["server_unit"],
        )


def test_list_images_finds_nested_session_folders(tmp_path):
    images = tmp_path / "images"
    make_frame(images / "session_a" / "capture_00001.png")
    make_frame(images / "session_b" / "capture_00002.png")
    make_frame(images / "capture_00003.png")
    (images / "notes.txt").write_text("not an image", encoding="utf-8")

    found = sorted(path.name for path in list_images(images))
    assert found == ["capture_00001.png", "capture_00002.png", "capture_00003.png"]

    flat = sorted(path.name for path in list_images(images, recursive=False))
    assert flat == ["capture_00003.png"]


def test_build_split_handles_nested_images(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    for index in range(6):
        session = images / f"session_{index % 2}"
        make_frame(session / f"capture_{index:05d}.png")
        write_label(labels / f"capture_{index:05d}.txt")

    stats = build_split(
        images,
        labels,
        tmp_path / "data",
        class_names=["server_unit"],
        val_ratio=0.34,
        seed=0,
    )
    assert stats.train + stats.val == 6
    assert stats.unlabeled == 0
    assert (
        len(list((tmp_path / "data" / "labels" / "train").glob("*.txt"))) == stats.train
    )


def test_read_label_classes_ignores_malformed_lines(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n\nbad line\n1 0.1 0.1 0.1 0.1\n", encoding="utf-8")
    assert read_label_classes(label) == [0, 1]
    assert read_label_classes(tmp_path / "missing.txt") == []


def test_yolo_lines_from_detections_are_normalised():
    from huanpan_perception.dataset import yolo_lines_from_detections
    from huanpan_perception.detector import Detection

    detections = [Detection(0, "server_unit", 0.9, (0.0, 0.0, 64.0, 48.0))]
    lines = yolo_lines_from_detections(detections, 128, 96)
    assert lines == ["0 0.250000 0.250000 0.500000 0.500000"]


def test_write_detection_labels_skips_existing(tmp_path):
    from huanpan_perception.dataset import write_detection_labels
    from huanpan_perception.detector import Detection

    labels = tmp_path / "labels"
    image = tmp_path / "images" / "a.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"")
    detections = [Detection(0, "server_unit", 0.9, (0.0, 0.0, 64.0, 48.0))]

    assert write_detection_labels(labels, image, detections, 128, 96) is True
    assert (labels / "a.txt").read_text().startswith("0 ")

    (labels / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    assert (
        write_detection_labels(labels, image, detections, 128, 96, skip_existing=True)
        is False
    )
    assert (labels / "a.txt").read_text() == "0 0.5 0.5 0.1 0.1\n"


def test_write_detection_labels_writes_empty_file_for_no_detection(tmp_path):
    """Empty means 'not labelled yet', so build_split skips it."""
    from huanpan_perception.dataset import write_detection_labels

    labels = tmp_path / "labels"
    image = tmp_path / "images" / "b.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"")

    assert write_detection_labels(labels, image, [], 128, 96) is True
    assert (labels / "b.txt").read_text(encoding="utf-8") == ""


def test_group_key_strips_the_frame_index():
    from huanpan_perception.dataset import group_key

    assert group_key(Path("clip01_000123.png")) == "clip01"
    assert group_key(Path("clipA-0007.jpg")) == "clipA"
    assert group_key(Path("capture_00001.png")) == "capture"
    # A name with no trailing index is its own group.
    assert group_key(Path("001_Color.png")) == "001_Color"
    assert group_key(Path("7.png")) == "7"


def test_auto_split_groups_frames_from_the_same_clip(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    for clip in ("clip01", "clip02", "clip03", "clip04"):
        for index in range(5):
            name = f"{clip}_{index:05d}"
            make_frame(images / f"{name}.png")
            write_label(labels / f"{name}.txt")

    stats = build_split(
        images, labels, tmp_path / "data", class_names=["server_unit"], val_ratio=0.25, seed=0
    )
    assert stats.split_mode == "group"
    assert stats.groups == {"clip01": 5, "clip02": 5, "clip03": 5, "clip04": 5}

    train_clips = {p.name.split("_")[0] for p in list_images(tmp_path / "data" / "images" / "train")}
    val_clips = {p.name.split("_")[0] for p in list_images(tmp_path / "data" / "images" / "val")}
    assert not (train_clips & val_clips), "a clip leaked across train and val"
    assert stats.train > 0 and stats.val > 0


def test_auto_split_falls_back_to_random_for_unique_names(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    for index in range(10):
        name = f"{index:03d}_Color"
        make_frame(images / f"{name}.png")
        write_label(labels / f"{name}.txt")

    stats = build_split(
        images, labels, tmp_path / "data", class_names=["server_unit"], val_ratio=0.2, seed=0
    )
    assert stats.split_mode == "random"


def test_group_split_with_a_single_group_is_an_error(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    for index in range(5):
        name = f"capture_{index:05d}"
        make_frame(images / f"{name}.png")
        write_label(labels / f"{name}.txt")

    with pytest.raises(ValueError, match="only one source group"):
        build_split(
            images,
            labels,
            tmp_path / "data",
            class_names=["server_unit"],
            group_by="group",
        )


def test_group_split_all_same_prefix_falls_back(tmp_path):
    """One group -> auto must not silently make val empty."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    for index in range(6):
        name = f"capture_{index:05d}"
        make_frame(images / f"{name}.png")
        write_label(labels / f"{name}.txt")

    stats = build_split(
        images, labels, tmp_path / "data", class_names=["server_unit"], val_ratio=0.34, seed=0
    )
    assert stats.split_mode == "random"
    assert stats.train + stats.val == 6
