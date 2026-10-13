"""Generate the optional AprilTag 36h11 reference/calibration board."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/perception/calibration/apriltag_3x2.png")
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--marker-mm", type=float, default=40.0)
    parser.add_argument("--separation-mm", type=float, default=10.0)
    parser.add_argument("--margin-mm", type=float, default=10.0)
    parser.add_argument("--pixels-per-mm", type=float, default=10.0)
    return parser.parse_args(argv)


def generate_board(
    *,
    columns: int = 3,
    rows: int = 2,
    marker_mm: float = 40.0,
    separation_mm: float = 10.0,
    margin_mm: float = 10.0,
    pixels_per_mm: float = 10.0,
) -> tuple[np.ndarray, dict]:
    if columns <= 0 or rows <= 0 or marker_mm <= 0 or separation_mm < 0:
        raise ValueError("invalid board geometry")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    board = cv2.aruco.GridBoard(
        (columns, rows),
        marker_mm / 1000.0,
        separation_mm / 1000.0,
        dictionary,
    )
    inner_width_mm = columns * marker_mm + (columns - 1) * separation_mm
    inner_height_mm = rows * marker_mm + (rows - 1) * separation_mm
    inner_size = (
        int(round(inner_width_mm * pixels_per_mm)),
        int(round(inner_height_mm * pixels_per_mm)),
    )
    inner = board.generateImage(inner_size, marginSize=0, borderBits=1)
    margin_px = int(round(margin_mm * pixels_per_mm))
    image = np.full(
        (inner.shape[0] + 2 * margin_px, inner.shape[1] + 2 * margin_px),
        255,
        dtype=np.uint8,
    )
    row_slice = slice(margin_px, margin_px + inner.shape[0])
    column_slice = slice(margin_px, margin_px + inner.shape[1])
    image[row_slice, column_slice] = inner
    metadata = {
        "dictionary": "DICT_APRILTAG_36h11",
        "columns": columns,
        "rows": rows,
        "marker_length_mm": marker_mm,
        "marker_separation_mm": separation_mm,
        "margin_mm": margin_mm,
        "print_width_mm": inner_width_mm + 2 * margin_mm,
        "print_height_mm": inner_height_mm + 2 * margin_mm,
        "marker_ids": list(range(columns * rows)),
        "pixels_per_mm": pixels_per_mm,
    }
    return image, metadata


def main(argv=None) -> int:
    args = parse_args(argv)
    image, metadata = generate_board(
        columns=args.columns,
        rows=args.rows,
        marker_mm=args.marker_mm,
        separation_mm=args.separation_mm,
        margin_mm=args.margin_mm,
        pixels_per_mm=args.pixels_per_mm,
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    dpi = float(args.pixels_per_mm) * 25.4
    Image.fromarray(image).save(output, dpi=(dpi, dpi))
    metadata_path = output.with_suffix(".yaml")
    metadata_path.write_text(
        yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(
        f"wrote {output} ({metadata['print_width_mm']} x "
        f"{metadata['print_height_mm']} mm at 100%)"
    )
    print(f"wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
