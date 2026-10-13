"""Depth-quality report for captured RGB-D frames.

Answers the question that decides the whole fine-localisation route: **is the
depth actually usable on the target, or is it full of holes?**

For every frame it reports the fraction of valid depth pixels overall and, when
labels are supplied, inside the annotated box. It also measures how flat the
target surface is, and writes a montage of colour + depth + validity so you can
see the holes rather than just read a number.

Usage::

    analyze_depth --images <dataset>/images --depth <dataset>/depth \\
                  --labels <dataset>/labels --output depth_report.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="colour image directory")
    parser.add_argument("--depth", default="", help="depth directory (default: sibling 'depth')")
    parser.add_argument("--labels", default="", help="optional YOLO label directory")
    parser.add_argument("--output", default="outputs/perception/depth_report.png")
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--max-tiles", type=int, default=12)
    return parser.parse_args(argv)


def load_depth(path: Path) -> np.ndarray:
    """Load a depth PNG as metres.

    RealSense writes ``16UC1`` PNGs in millimetres; float images are assumed to
    already be metres.
    """
    array = np.asarray(Image.open(path))
    # Pillow versions differ here: a 16-bit PNG may be exposed as uint16 or
    # promoted to mode ``I``/int32.  Both still contain RealSense millimetres.
    if array.dtype.kind in "iu" and array.dtype.itemsize > 1:
        return array.astype(np.float32) / 1000.0
    return array.astype(np.float32)


def label_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    if not label_path.is_file():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = (float(v) for v in parts)
        boxes.append(
            (
                max(0, int((cx - w / 2) * width)),
                max(0, int((cy - h / 2) * height)),
                min(width, int((cx + w / 2) * width)),
                min(height, int((cy + h / 2) * height)),
            )
        )
    return boxes


def depth_colour(depth_m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    span = float(depth_m[valid].max()) if valid.any() else 1.0
    span = max(span, 0.3)
    scaled = np.clip(255.0 * (1.0 - depth_m / span), 0, 255).astype(np.uint8)
    rgb = np.repeat(scaled[:, :, None], 3, axis=2)
    rgb[~valid] = (255, 0, 0)  # holes in red
    return rgb


def main(argv=None) -> int:
    from .launcher import ensure_runtime

    ensure_runtime("huanpan_perception.analyze_depth")

    args = parse_args(argv)
    images_dir = Path(args.images).expanduser()
    depth_dir = Path(args.depth).expanduser() if args.depth else images_dir.parent / "depth"
    labels_dir = Path(args.labels).expanduser() if args.labels else None

    if not depth_dir.is_dir():
        print(f"[analyze_depth] depth directory not found: {depth_dir}")
        print("  capture frames with save_depth:=true to produce one")
        return 2

    images = sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print(f"[analyze_depth] no colour images in {images_dir}")
        return 2

    records = []
    tiles = []
    for image_path in images:
        depth_path = depth_dir / f"{image_path.stem}.png"
        if not depth_path.is_file():
            records.append({"image": image_path.name, "error": "no depth file"})
            continue

        depth = load_depth(depth_path)
        valid = np.isfinite(depth) & (depth > args.min_depth_m) & (depth < args.max_depth_m)

        record = {
            "image": image_path.name,
            "depth_size": [int(depth.shape[1]), int(depth.shape[0])],
            "valid_ratio": float(valid.mean()),
            "median_depth_m": float(np.median(depth[valid])) if valid.any() else None,
        }

        if labels_dir is not None:
            height, width = depth.shape[:2]
            for index, (x0, y0, x1, y1) in enumerate(
                label_boxes(labels_dir / f"{image_path.stem}.txt", width, height)
            ):
                window = valid[y0:y1, x0:x1]
                if window.size == 0:
                    continue
                values = depth[y0:y1, x0:x1][window]
                record[f"box{index}_valid_ratio"] = float(window.mean())
                if values.size:
                    record[f"box{index}_median_depth_m"] = float(np.median(values))
                    # Spread of depth on the target: a flat face has a small
                    # spread relative to its distance.
                    record[f"box{index}_depth_std_m"] = float(values.std())
                    record[f"box{index}_depth_mad_m"] = float(
                        np.median(np.abs(values - np.median(values)))
                    )
        records.append(record)

        if len(tiles) < args.max_tiles:
            colour = np.asarray(Image.open(image_path).convert("RGB"))
            depth_rgb = depth_colour(depth, valid)
            if depth_rgb.shape[:2] != colour.shape[:2]:
                depth_rgb = np.asarray(
                    Image.fromarray(depth_rgb).resize((colour.shape[1], colour.shape[0]))
                )
            tiles.append(np.hstack([colour, depth_rgb]))

    ok = [r for r in records if "valid_ratio" in r]
    box_ratios = [
        r[key] for r in ok for key in r if key.endswith("_valid_ratio") and key != "valid_ratio"
    ]
    summary = {
        "frames": len(records),
        "frames_with_depth": len(ok),
        "overall_valid_ratio": float(np.mean([r["valid_ratio"] for r in ok])) if ok else None,
        "box_valid_ratio": float(np.mean(box_ratios)) if box_ratios else None,
        "per_frame": records,
    }

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.parent / f"{output.stem}_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if tiles:
        columns = 2
        rows = (len(tiles) + columns - 1) // columns
        th, tw = tiles[0].shape[:2]
        canvas = np.full((rows * th + (rows + 1) * 4, columns * tw + (columns + 1) * 4, 3), 20, np.uint8)
        for index, tile in enumerate(tiles):
            row, col = divmod(index, columns)
            y = 4 + row * (th + 4)
            x = 4 + col * (tw + 4)
            canvas[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
        Image.fromarray(canvas).save(output)

    print(f"帧数                : {summary['frames']}（有深度 {summary['frames_with_depth']}）")
    if summary["overall_valid_ratio"] is not None:
        print(f"全图深度有效率      : {summary['overall_valid_ratio'] * 100:.1f}%")
    if summary["box_valid_ratio"] is not None:
        print(f"目标框内深度有效率  : {summary['box_valid_ratio'] * 100:.1f}%")
        verdict = (
            "✅ 深度可用于反投影"
            if summary["box_valid_ratio"] > 0.6
            else "⚠️ 深度空洞较多，需走平面拟合或参数化模型"
            if summary["box_valid_ratio"] > 0.25
            else "❌ 目标上基本没有深度，必须用 CAD/参数化模型 + PnP"
        )
        print(f"结论                : {verdict}")
    print(f"报告                : {report_path}")
    if tiles:
        print(f"可视化（彩色 | 深度，红色=空洞）: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
