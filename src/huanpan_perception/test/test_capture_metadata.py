"""Dataset naming and metadata tests."""

from pathlib import Path

from huanpan_perception.capture_metadata import next_capture_index, stamp_to_ns


class Stamp:
    sec = 12
    nanosec = 345


def test_stamp_to_ns():
    assert stamp_to_ns(Stamp()) == 12_000_000_345


def test_next_capture_index_never_overwrites_existing_pairs(tmp_path: Path):
    images = tmp_path / "images"
    depth = tmp_path / "depth"
    images.mkdir()
    depth.mkdir()
    (images / "fineA_00035.png").write_bytes(b"")
    (depth / "fineA_00034.png").write_bytes(b"")
    (images / "other_99999.png").write_bytes(b"")
    assert next_capture_index(images, depth, "fineA") == 36


def test_next_capture_index_starts_at_one(tmp_path: Path):
    assert next_capture_index(tmp_path / "images", tmp_path / "depth", "new") == 1
