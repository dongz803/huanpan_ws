"""Dataset layout helpers for the real-image YOLO workflow.

Canonical layout produced by :mod:`huanpan_perception.capture_node` and
:mod:`huanpan_perception.label_cli`::

    <root>/
      images/            captured frames (flat)
      labels/            one YOLO .txt per frame (flat, same stem)
      data/              generated split for training
        images/{train,val}
        labels/{train,val}
      data.yaml

``build_split`` turns the flat ``images`` + ``labels`` pair into the split that
Ultralytics expects and writes ``data.yaml``.
"""

from __future__ import annotations

import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

_TRAILING_INDEX = re.compile(r"[_-]?\d+$")


def group_key(image_path: Path) -> str:
    """Source-group a frame belongs to, derived from its file name.

    Frames extracted from one video (``clip01_000123.png``) or captured in one
    session share a group. Splitting train/val **by group** keeps near-identical
    neighbouring frames out of the validation set, which would otherwise inflate
    the metrics.
    """
    stem = image_path.stem
    return _TRAILING_INDEX.sub("", stem) or stem


@dataclass
class SplitStats:
    train: int
    val: int
    unlabeled: int
    class_counts: dict[int, int]
    data_yaml: Path
    split_mode: str = "random"
    groups: dict[str, int] = None  # type: ignore[assignment]


def list_images(directory: Path, recursive: bool = True) -> list[Path]:
    """List image files under ``directory``.

    Recursive by default so that photos organised into per-session
    sub-directories (``images/2026-09-15/...``) are still found. Label files are
    written beside a flat ``labels/`` directory keyed by file stem, so **stems
    must be unique** across sub-directories; the capture node already names
    frames ``<prefix>_00001.png`` which guarantees that.
    """
    if not directory.is_dir():
        return []
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def label_path_for(labels_dir: Path, image_path: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def read_label_classes(label_path: Path) -> list[int]:
    if not label_path.is_file():
        return []
    classes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            classes.append(int(float(line.split()[0])))
        except (ValueError, IndexError):
            continue
    return classes


def count_classes(label_files: Iterable[Path]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for label_file in label_files:
        for class_id in read_label_classes(label_file):
            counts[class_id] = counts.get(class_id, 0) + 1
    return counts


def write_data_yaml(
    path: Path,
    dataset_root: Path,
    class_names: Sequence[str],
) -> None:
    lines = [
        f"path: {dataset_root.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(class_names)],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def yolo_lines_from_detections(detections, width: int, height: int) -> list[str]:
    """Convert detector output into YOLO ``class cx cy w h`` lines."""
    lines = []
    for det in detections:
        x0, y0, x1, y1 = det.xyxy
        lines.append(
            f"{det.class_id} "
            f"{(x0 + x1) / 2.0 / width:.6f} {(y0 + y1) / 2.0 / height:.6f} "
            f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}"
        )
    return lines


def write_detection_labels(
    labels_dir: Path,
    image_path: Path,
    detections,
    width: int,
    height: int,
    *,
    skip_existing: bool = False,
) -> bool:
    """Write predicted boxes as a YOLO label file. Returns True if written.

    Used for human-in-the-loop pre-annotation: a first model labels the bulk of
    the images, then a human only corrects them in ``label_dataset``.

    Images with no detection get an **empty** file. ``build_split`` treats an
    empty label as "not labelled yet" and skips it, so nothing is silently
    trained as a background/negative sample.
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    target = label_path_for(labels_dir, image_path)
    if skip_existing and target.exists():
        return False
    lines = yolo_lines_from_detections(detections, width, height)
    target.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )
    return True


def build_split(
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    *,
    class_names: Sequence[str],
    val_ratio: float = 0.2,
    seed: int = 0,
    require_label: bool = True,
    group_by: str = "auto",
) -> SplitStats:
    """Copy labelled frames into ``out_dir/images|labels/{train,val}``.

    ``group_by`` controls how frames are assigned to the validation set:

    ``auto`` (default)
        Split by source group (see :func:`group_key`) when the frames come from
        more than one group. Frames extracted from a single video are almost
        identical to their neighbours, so a random split would put a near-copy
        of every validation frame into the training set and report an
        optimistically wrong mAP. Falls back to ``random`` when there is only
        one group.
    ``group``
        Always split by group; raises if only one group exists.
    ``random``
        Plain random split. Fine for independently captured photos with unique
        names (``001_Color.png`` ...).
    """
    images = list_images(images_dir)
    if not images:
        raise FileNotFoundError(f"no images found in {images_dir}")

    labeled: list[Path] = []
    unlabeled = 0
    for image in images:
        label = label_path_for(labels_dir, image)
        classes = read_label_classes(label)
        if classes or not require_label:
            labeled.append(image)
        else:
            unlabeled += 1
    if not labeled:
        raise FileNotFoundError(
            f"no labelled frames found: every image in {images_dir} is missing "
            f"a matching .txt in {labels_dir}"
        )

    grouped: dict[str, list[Path]] = {}
    for image in labeled:
        grouped.setdefault(group_key(image), []).append(image)
    group_sizes = {name: len(files) for name, files in grouped.items()}

    mode = group_by
    if mode == "auto":
        # Group splitting only works, and only matters, when there are several
        # sources AND at least one of them contributes more than one frame. A
        # single clip cannot be split at all, so fall back to random rather than
        # refusing to run (train_cli warns that the metrics may be optimistic).
        has_multi_frame_group = any(len(files) > 1 for files in grouped.values())
        mode = "group" if len(grouped) > 1 and has_multi_frame_group else "random"
    if mode == "group" and len(grouped) < 2:
        raise ValueError(
            "cannot split by group: only one source group found "
            f"({next(iter(grouped))!r}). Name frames per clip, e.g. "
            "clip01_00001.png, or pass group_by='random'."
        )

    rng = random.Random(seed)
    if mode == "group":
        names = sorted(grouped)
        rng.shuffle(names)
        target_val = max(1, int(round(len(labeled) * val_ratio)))
        val_names: list[str] = []
        running = 0
        for name in names:
            if running >= target_val and val_names:
                break
            val_names.append(name)
            running += group_sizes[name]
        if len(val_names) == len(names):  # never send everything to val
            val_names = val_names[: max(1, len(names) - 1)]
        val_set = {image for name in val_names for image in grouped[name]}
        splits = {
            "val": [image for image in labeled if image in val_set],
            "train": [image for image in labeled if image not in val_set],
        }
    else:
        shuffled = list(labeled)
        rng.shuffle(shuffled)
        val_count = (
            max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
        )
        splits = {"val": shuffled[:val_count], "train": shuffled[val_count:]}

    for split, files in splits.items():
        image_target = out_dir / "images" / split
        label_target = out_dir / "labels" / split
        image_target.mkdir(parents=True, exist_ok=True)
        label_target.mkdir(parents=True, exist_ok=True)
        for image in files:
            shutil.copy2(image, image_target / image.name)
            label = label_path_for(labels_dir, image)
            if label.is_file():
                shutil.copy2(label, label_target / label.name)

    data_yaml = out_dir / "data.yaml"
    write_data_yaml(data_yaml, out_dir, class_names)

    train_labels = [out_dir / "labels" / "train" / f"{p.stem}.txt" for p in splits["train"]]
    val_labels = [out_dir / "labels" / "val" / f"{p.stem}.txt" for p in splits["val"]]
    return SplitStats(
        train=len(splits["train"]),
        val=len(splits["val"]),
        unlabeled=unlabeled,
        class_counts=count_classes(train_labels + val_labels),
        data_yaml=data_yaml,
        split_mode=mode,
        groups=group_sizes,
    )
