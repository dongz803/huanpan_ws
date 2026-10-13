"""Interactive labelling logic tests.

These cover the draw → save → next-image flow without opening a window, which
is where the "cannot draw a box" regression lived: the render loop reloaded the
frame on every iteration and that reload also wiped the box being dragged.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")

from huanpan_perception.label_cli import Annotator, parse_args  # noqa: E402


def make_annotator(tmp_path, names=("a.png", "b.png")) -> Annotator:
    images = tmp_path / "images"
    images.mkdir()
    for index, name in enumerate(names):
        array = np.full((80, 120, 3), 100 + index * 40, dtype=np.uint8)
        Image.fromarray(array).save(images / name)
    return Annotator(parse_args(["--images", str(images)]))


def drag(annotator: Annotator, x0: int, y0: int, x1: int, y1: int) -> None:
    annotator.on_mouse(cv2.EVENT_LBUTTONDOWN, x0, y0, 0, None)
    annotator.on_mouse(cv2.EVENT_MOUSEMOVE, x1, y1, 0, None)
    annotator.on_mouse(cv2.EVENT_LBUTTONUP, x1, y1, 0, None)


def test_drag_creates_a_box(tmp_path):
    annotator = make_annotator(tmp_path)
    annotator.on_mouse(cv2.EVENT_LBUTTONDOWN, 10, 20, 0, None)
    assert annotator.dragging
    assert annotator.box == (10, 20, 10, 20)

    annotator.on_mouse(cv2.EVENT_MOUSEMOVE, 90, 60, 0, None)
    assert annotator.box == (10, 20, 90, 60)

    annotator.on_mouse(cv2.EVENT_LBUTTONUP, 90, 60, 0, None)
    assert not annotator.dragging
    assert annotator.box == (10, 20, 90, 60)


def test_mouse_move_without_press_does_nothing(tmp_path):
    annotator = make_annotator(tmp_path)
    annotator.on_mouse(cv2.EVENT_MOUSEMOVE, 50, 50, 0, None)
    assert annotator.box is None


def test_redrawing_replaces_the_previous_box(tmp_path):
    annotator = make_annotator(tmp_path)
    drag(annotator, 10, 10, 50, 50)
    drag(annotator, 20, 20, 100, 70)
    assert annotator.box == (20, 20, 100, 70)


def test_box_survives_repeated_frame_loads(tmp_path):
    """Regression: loading the frame used to reset the in-progress box.

    The render loop calls ``load_image`` every ~20 ms, so any reset there made
    the box invisible and made ``s`` report "no box drawn".
    """
    annotator = make_annotator(tmp_path)
    drag(annotator, 10, 20, 90, 60)

    for _ in range(5):
        annotator.load_image(annotator.index)  # what the render loop does
        assert annotator.box == (10, 20, 90, 60), "box was wiped by a frame load"


def test_save_writes_a_normalised_yolo_line(tmp_path):
    annotator = make_annotator(tmp_path)
    image = annotator.load_image(0)
    annotator.box = (0, 0, 120, 80)  # the whole frame
    annotator.save(0, image)

    text = (annotator.labels_dir / "a.txt").read_text(encoding="utf-8").strip()
    parts = text.split()
    assert parts[0] == "0"
    assert float(parts[1]) == pytest.approx(0.5)
    assert float(parts[2]) == pytest.approx(0.5)
    assert float(parts[3]) == pytest.approx(1.0)
    assert float(parts[4]) == pytest.approx(1.0)


def test_save_without_a_box_writes_nothing(tmp_path):
    annotator = make_annotator(tmp_path)
    annotator.box = None
    annotator.save(0, annotator.load_image(0))
    assert not (annotator.labels_dir / "a.txt").exists()


def test_save_then_reload_is_lossless(tmp_path):
    annotator = make_annotator(tmp_path)
    image = annotator.load_image(0)
    for box in [(10, 20, 90, 60), (0, 0, 120, 80), (7, 13, 99, 61)]:
        annotator.box = box
        annotator.save(0, image)
        assert annotator.load_box(0, image) == box


def test_each_image_keeps_its_own_box(tmp_path):
    annotator = make_annotator(tmp_path)
    annotator.box = (10, 10, 50, 50)
    annotator.save(0, annotator.load_image(0))
    annotator.box = (20, 20, 60, 60)
    annotator.save(1, annotator.load_image(1))

    assert annotator.load_box(0, annotator.load_image(0)) == (10, 10, 50, 50)
    assert annotator.load_box(1, annotator.load_image(1)) == (20, 20, 60, 60)


def test_delete_removes_the_label_and_the_box(tmp_path):
    annotator = make_annotator(tmp_path)
    annotator.box = (10, 10, 50, 50)
    annotator.save(0, annotator.load_image(0))
    assert (annotator.labels_dir / "a.txt").exists()

    annotator.delete()
    assert not (annotator.labels_dir / "a.txt").exists()
    assert annotator.box is None


def test_missing_label_loads_as_no_box(tmp_path):
    annotator = make_annotator(tmp_path)
    assert annotator.load_box(0, annotator.load_image(0)) is None


def test_loading_a_non_image_raises(tmp_path):
    annotator = make_annotator(tmp_path)
    (annotator.labels_dir).mkdir(parents=True, exist_ok=True)
    broken = annotator.images[0].parent / "broken.png"
    broken.write_text("not an image", encoding="utf-8")
    annotator.images = [broken]
    with pytest.raises(RuntimeError):
        annotator.load_image(0)
