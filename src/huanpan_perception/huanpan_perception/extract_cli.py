"""Extract training frames from recorded videos.

Extracting every frame of a clip produces fifty nearly identical images: the
model learns almost nothing extra and, worse, a random train/val split leaks
near-copies of the validation frames into training. This tool therefore:

* samples at a time interval rather than every frame;
* skips a frame that is nearly identical to the last one **kept** (cheap
  downscaled-grayscale comparison), so a person standing still does not create
  twenty duplicates;
* names frames ``<clip>_<index>.png`` so :func:`build_split` can keep whole
  clips on one side of the split.

For RealSense ``.bag`` recordings, convert first::

    rs-convert -i recording.bag -p output_prefix

Usage::

    python -m huanpan_perception.extract_cli \
        --video clips/ --output outputs/perception/datasets/rack_grid_v1/images \
        --every-seconds 0.7
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="video file or directory")
    parser.add_argument("--output", required=True, help="destination images directory")
    parser.add_argument(
        "--every-seconds",
        type=float,
        default=0.7,
        help="minimum time between kept frames (default: %(default)s)",
    )
    parser.add_argument(
        "--min-difference",
        type=float,
        default=0.012,
        help="skip a frame whose mean absolute difference to the last kept "
        "frame is below this (0..1); 0 disables the check",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    return parser.parse_args(argv)


def sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_") or "clip"


def signature(frame: np.ndarray) -> np.ndarray:
    """Small grayscale fingerprint used for the near-duplicate test."""
    import cv2

    small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def collect_videos(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def extract_video(
    video: Path,
    output_dir: Path,
    *,
    every_seconds: float,
    min_difference: float,
    max_frames: int,
    jpeg_quality: int,
    dry_run: bool,
) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return {"video": str(video), "error": "cannot open"}

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 1e-6:
        fps = 30.0
    step = max(1, int(round(fps * max(every_seconds, 1e-3))))

    clip = sanitize(video.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    scanned = 0
    skipped_similar = 0
    last_signature: np.ndarray | None = None
    index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        scanned += 1
        if index % step:
            index += 1
            continue
        index += 1

        if min_difference > 0 and last_signature is not None:
            difference = float(
                np.abs(signature(frame) - last_signature).mean() / 255.0
            )
            if difference < min_difference:
                skipped_similar += 1
                continue

        kept += 1
        if not dry_run:
            height, width = frame.shape[:2]
            name = f"{clip}_{kept:05d}"
            if frame.shape[2] == 3:
                path = output_dir / f"{name}.jpg"
                cv2.imwrite(
                    str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                )
            else:  # pragma: no cover - defensive
                path = output_dir / f"{name}.png"
                cv2.imwrite(str(path), frame)
        last_signature = signature(frame)

        if max_frames and kept >= max_frames:
            break

    capture.release()
    return {
        "video": str(video),
        "clip": clip,
        "fps": round(fps, 2),
        "frames_in_video": total,
        "scanned": scanned,
        "kept": kept,
        "skipped_similar": skipped_similar,
        "frame_step": step,
    }


def main(argv=None) -> int:
    from .launcher import ensure_runtime

    ensure_runtime("huanpan_perception.extract_cli")

    args = parse_args(argv)
    source = Path(args.video).expanduser()
    if not source.exists():
        print(f"[extract] not found: {source}")
        return 2
    videos = collect_videos(source)
    if not videos:
        print(f"[extract] no video files under {source}")
        return 2

    output_dir = Path(args.output).expanduser()
    results = []
    for video in videos:
        result = extract_video(
            video,
            output_dir,
            every_seconds=args.every_seconds,
            min_difference=args.min_difference,
            max_frames=args.max_frames,
            jpeg_quality=args.jpeg_quality,
            dry_run=args.dry_run,
        )
        results.append(result)
        if "error" in result:
            print(f"[extract] {video.name}: {result['error']}")
        else:
            print(
                f"[extract] {video.name}: kept {result['kept']} of "
                f"{result['scanned']} scanned frames "
                f"({result['skipped_similar']} dropped as near-duplicates, "
                f"every {result['frame_step']} frame(s))"
            )

    total = sum(item.get("kept", 0) for item in results)
    print(f"[extract] {total} frames -> {output_dir}")
    print(
        "[extract] 下一步：label_dataset --images "
        f"{output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
