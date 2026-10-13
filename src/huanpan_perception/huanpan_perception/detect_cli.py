"""Offline detection CLI: run the trained detector on images or a folder.

Useful before the D435 driver is installed, and for checking a freshly trained
model against captured footage::

    python -m huanpan_perception.detect_cli \
        --model runs/detect/server_unit/weights/best.pt \
        --source outputs/perception/captures/handheld/images \
        --output outputs/perception/predictions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import write_detection_labels
from .detector import DetectorError, RackDetector
from .viz import draw_detections, draw_label_boxes, montage

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="trained .pt weights")
    parser.add_argument("--source", required=True, help="image file or directory")
    parser.add_argument("--output", default="outputs/perception/predictions")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-images", type=int, default=0, help="0 = all")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument(
        "--write-labels",
        default="",
        help="write predicted boxes as YOLO .txt into this directory "
        "(human-in-the-loop pre-annotation)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="with --write-labels: keep label files that already exist",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="label directory: also draw the ground truth in green so you can "
        "see how close the prediction is",
    )
    parser.add_argument(
        "--montage",
        default="",
        help="write a single overview image of all annotated frames here",
    )
    parser.add_argument("--montage-columns", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=470)
    return parser.parse_args(argv)


def collect_images(source: Path, limit: int) -> list[Path]:
    if source.is_file():
        return [source]
    images = sorted(
        path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    return images[:limit] if limit else images


def main(argv=None) -> int:
    from .launcher import ensure_runtime

    ensure_runtime("huanpan_perception.detect_cli")

    args = parse_args(argv)
    source = Path(args.source).expanduser()
    if not source.exists():
        print(f"[detect_cli] source not found: {source}")
        return 2

    images = collect_images(source, args.max_images)
    if not images:
        print(f"[detect_cli] no images found under {source}")
        return 2

    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        detector = RackDetector(
            args.model,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
        ).load()
    except DetectorError as exc:
        print(f"[detect_cli] {exc}")
        return 2

    summary = []
    labels_dir = Path(args.write_labels).expanduser() if args.write_labels else None
    gt_dir = Path(args.labels).expanduser() if args.labels else None
    tiles = []
    written = 0
    empty = 0
    for index, path in enumerate(images, start=1):
        frame = np.asarray(Image.open(path).convert("RGB"))
        detections = detector.detect(frame)
        if gt_dir is not None:
            annotated = draw_label_boxes(
                frame, gt_dir / f"{path.stem}.txt", "GT"
            )
            annotated = draw_detections(
                annotated, detections, color=(255, 70, 70), width=4
            )
        else:
            annotated = draw_detections(frame, detections, crosshair=True)
        Image.fromarray(annotated).save(output_dir / f"{path.stem}_det.png")

        if args.montage:
            picture = Image.fromarray(annotated)
            scale = args.tile_width / picture.width
            tiles.append(
                np.asarray(
                    picture.resize(
                        (args.tile_width, max(1, int(picture.height * scale)))
                    )
                )
            )

        if labels_dir is not None:
            wrote = write_detection_labels(
                labels_dir,
                path,
                detections,
                frame.shape[1],
                frame.shape[0],
                skip_existing=args.skip_existing,
            )
            if wrote:
                written += 1
                if not detections:
                    empty += 1

        summary.append(
            {
                "image": str(path),
                "detections": [
                    {
                        "class_name": d.class_name,
                        "score": round(d.confidence, 4),
                        "xyxy": [round(v, 2) for v in d.xyxy],
                        "center": [round(v, 2) for v in d.center],
                    }
                    for d in detections
                ],
            }
        )
        if index % 10 == 0 or index == len(images):
            print(f"[detect_cli] {index}/{len(images)} images")

    if args.save_json:
        (output_dir / "detections.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    if args.montage and tiles:
        canvas = montage(tiles, columns=args.montage_columns)
        montage_path = Path(args.montage).expanduser()
        montage_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(canvas).save(montage_path)
        print(f"[detect_cli] overview image -> {montage_path}")
    print(f"[detect_cli] wrote {len(images)} annotated images to {output_dir}")
    if labels_dir is not None:
        print(
            f"[detect_cli] pre-annotated {written}/{len(images)} frames into "
            f"{labels_dir}"
            + (f" ({empty} had no detection and need a hand-drawn box)" if empty else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
