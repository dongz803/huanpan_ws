"""Build a labelled slot-detection dataset from the MuJoCo workcell.

The rack geometry is known exactly in simulation, so every image comes with a
free pixel-accurate bounding box, the slot index and the slot's 6D pose in the
world frame. That is the ground truth the perception module will later have to
recover from a real Intel RealSense D435.

Usage::

    python -m slotvision.generate --out outputs/perception/datasets/slot_detect_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .labels import (
    detection_to_dict,
    draw_detections,
    montage,
    write_data_yaml,
    write_labels,
    write_meta,
)
from .scene import SlotWorkcell

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    WORKSPACE_ROOT
    / "src"
    / "huanpan_description"
    / "mujoco"
    / "universal_robots_ur5e"
    / "workcell.xml"
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="dataset directory")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train", type=int, default=240)
    parser.add_argument("--val", type=int, default=60)
    parser.add_argument("--hard", type=int, default=40, help="wide-jitter test split")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="only label the 12 individual slots, skip the whole-grid box",
    )
    parser.add_argument(
        "--depth-samples",
        type=int,
        default=6,
        help="how many validation frames also store 16-bit depth PNGs",
    )
    parser.add_argument(
        "--preview-count", type=int, default=6, help="ground-truth preview frames"
    )
    parser.add_argument(
        "--augment",
        default="light",
        choices=["none", "light", "strong"],
        help="photometric augmentation strength",
    )
    return parser.parse_args(argv)


def augment(
    image: np.ndarray, rng: np.random.Generator, strength: str
) -> np.ndarray:
    """Photometric only: never moves geometry, so labels stay valid."""
    if strength == "none":
        return image
    scale = 1.0 if strength == "light" else 1.7
    data = image.astype(np.float32)

    gain = 1.0 + rng.uniform(-0.18, 0.18) * scale
    bias = rng.uniform(-18.0, 18.0) * scale
    data = np.clip(data * gain + bias, 0.0, 255.0)

    gamma = 1.0 + rng.uniform(-0.25, 0.25) * scale
    data = 255.0 * np.power(data / 255.0, gamma)

    data = data + rng.normal(0.0, 3.0 * scale, size=data.shape)
    return np.clip(data, 0.0, 255.0).astype(np.uint8)


def maybe_blur(
    image: np.ndarray, rng: np.random.Generator, strength: str
) -> np.ndarray:
    if strength == "none" or rng.random() > (0.2 if strength == "light" else 0.45):
        return image
    radius = rng.uniform(0.4, 1.1) * (1.0 if strength == "light" else 1.6)
    return np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius)))


def build_split(
    scene: SlotWorkcell,
    out_dir: Path,
    split: str,
    count: int,
    level: str,
    rng: np.random.Generator,
    include_grid: bool,
    augment_strength: str,
    depth_samples: int,
    preview_count: int,
) -> tuple[list[dict], list[Image.Image]]:
    image_dir = out_dir / "images" / split
    label_dir = out_dir / "labels" / split
    depth_dir = out_dir / "depth" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    if depth_samples:
        depth_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    previews: list[Image.Image] = []
    kept = 0
    attempts = 0
    max_attempts = count * 6

    while kept < count and attempts < max_attempts:
        attempts += 1
        scene.reset()
        scene_state = scene.randomize(rng, level=level)
        detections = scene.sample_detections(include_grid=include_grid)
        n_slots = sum(1 for det in detections if det.slot >= 0)
        if n_slots < 6:
            # Too much of the rack left the frame after randomisation.
            continue

        rgb = scene.render_rgb()
        rgb = augment(rgb, rng, augment_strength)
        rgb = maybe_blur(rgb, rng, augment_strength)

        name = f"{split}_{kept:05d}"
        Image.fromarray(rgb).save(image_dir / f"{name}.png")
        write_labels(label_dir / f"{name}.txt", detections, scene.width, scene.height)

        if kept < depth_samples:
            depth = scene.render_depth()
            depth_mm = np.nan_to_num(depth, nan=0.0, posinf=0.0) * 1000.0
            depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
            Image.fromarray(depth_mm).save(depth_dir / f"{name}.png")

        cam_pos, cam_quat = scene.camera_pose()
        records.append(
            {
                "image": f"images/{split}/{name}.png",
                "label": f"labels/{split}/{name}.txt",
                "split": split,
                "randomisation": level,
                "camera_pos": [round(float(v), 6) for v in cam_pos],
                "camera_quat_wxyz": [round(float(v), 6) for v in cam_quat],
                "scene": {
                    key: [round(float(v), 6) for v in value]
                    for key, value in scene_state.items()
                },
                "detections": [detection_to_dict(det) for det in detections],
            }
        )

        if len(previews) < preview_count:
            previews.append(draw_detections(rgb, detections))
        kept += 1

    if kept < count:
        print(
            f"warning: only generated {kept}/{count} {split} frames "
            f"({attempts} attempts)",
            file=sys.stderr,
        )
    return records, previews


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = args.out if args.out.is_absolute() else WORKSPACE_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    started = time.time()
    all_records: list[dict] = []
    preview_rows: list[Image.Image] = []
    intrinsics: dict = {}

    with SlotWorkcell(
        model_path=str(args.model),
        width=args.width,
        height=args.height,
    ) as scene:
        intrinsics = scene.intrinsics.to_dict()
        splits = [
            ("train", args.train, "train"),
            ("val", args.val, "train"),
            ("hard", args.hard, "hard"),
        ]
        for split, count, level in splits:
            depth_samples = args.depth_samples if split != "train" else 0
            records, previews = build_split(
                scene,
                out_dir,
                split,
                count,
                level,
                rng,
                include_grid=not args.no_grid,
                augment_strength=args.augment,
                depth_samples=depth_samples,
                preview_count=args.preview_count if split == "train" else 0,
            )
            all_records.extend(records)
            if split == "train":
                preview_rows.extend(previews)
            print(f"{split}: {len(records)} frames")

    write_meta(
        out_dir / "meta.json",
        intrinsics=intrinsics,
        class_names=SlotWorkcell.CLASS_NAMES,
        samples=all_records,
    )
    write_data_yaml(
        out_dir / "data.yaml",
        out_dir,
        SlotWorkcell.CLASS_NAMES,
        ["train", "val"],
    )

    if preview_rows:
        canvas = montage(preview_rows, columns=3)
        canvas.save(out_dir / "preview_ground_truth.png")
        print(f"ground-truth preview: {out_dir / 'preview_ground_truth.png'}")

    summary = {
        "dataset": str(out_dir),
        "counts": {"train": args.train, "val": args.val, "hard": args.hard},
        "intrinsics": intrinsics,
        "classes": list(SlotWorkcell.CLASS_NAMES),
        "elapsed_s": round(time.time() - started, 2),
    }
    (out_dir / "generate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
