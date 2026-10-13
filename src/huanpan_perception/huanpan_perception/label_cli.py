"""Minimal single-class bounding-box annotator (OpenCV window).

With one class (``server_unit``) labelling is just "drag one box, press s". This
tool iterates the captured frames and writes YOLO ``.txt`` files, so a few
hundred images take a few minutes.

Keys
----
    drag left button   draw a box (redrawing replaces the previous one)
    s                  save the box and go to the next image
    n / space          skip to the next image without saving
    d                  delete the existing label for this image
    b                  go back to the previous image
    q / Esc            quit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import IMAGE_SUFFIXES, label_path_for, list_images, read_label_classes


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="directory of captured frames")
    parser.add_argument("--labels", default="", help="label directory (default: sibling 'labels')")
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--class-name", default="server_unit")
    parser.add_argument("--start", type=int, default=0, help="index of the first image")
    parser.add_argument(
        "--preview",
        default="",
        help="do not open the GUI: render all existing labels into this image "
        "and exit (sanity check before training)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not open the GUI: report label consistency (box counts, "
        "aspect/area outliers) and write a montage of the worst offenders",
    )
    parser.add_argument(
        "--check-output",
        default="outputs/perception/label_outliers.png",
        help="where --check writes its montage",
    )
    return parser.parse_args(argv)


class Annotator:
    def __init__(self, args: argparse.Namespace) -> None:
        import cv2  # imported lazily so the module is importable without OpenCV

        self.cv2 = cv2
        self.images = list_images(Path(args.images).expanduser())
        if not self.images:
            raise FileNotFoundError(f"no images found in {args.images}")
        self.labels_dir = (
            Path(args.labels).expanduser()
            if args.labels
            else Path(args.images).expanduser().parent / "labels"
        )
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.class_id = int(args.class_id)
        self.class_name = args.class_name
        self.index = max(0, int(args.start))
        self.box: tuple[int, int, int, int] | None = None
        self.dragging = False
        self.origin = (0, 0)
        self.window = "huanpan label (drag=box  s=save  n=next  d=delete  b=back  q=quit)"

    # ------------------------------------------------------------------ logic
    def load_image(self, index: int) -> np.ndarray:
        """Read one frame. Does not touch the in-progress box."""
        image = self.cv2.imread(str(self.images[index]), self.cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read {self.images[index]}")
        return image

    def load_box(self, index: int, image: np.ndarray) -> tuple[int, int, int, int] | None:
        """Read the saved box for one frame, if any."""
        label = label_path_for(self.labels_dir, self.images[index])
        if not label.is_file():
            return None
        height, width = image.shape[:2]
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, w, h = (float(v) for v in parts)
            # round(), not int(): truncation loses a pixel and would make a box
            # drift every time it is loaded and saved again.
            return (
                round((cx - w / 2) * width),
                round((cy - h / 2) * height),
                round((cx + w / 2) * width),
                round((cy + h / 2) * height),
            )
        return None

    def load_current(self) -> np.ndarray:
        """Convenience for tests: read the frame and its saved box."""
        image = self.load_image(self.index)
        self.box = self.load_box(self.index, image)
        return image

    def save(self, index: int, image: np.ndarray) -> None:
        """Write the current box for ``index`` as a YOLO line."""
        if self.box is None:
            print("no box drawn; nothing saved")
            return
        height, width = image.shape[:2]
        x0, y0, x1, y1 = self._ordered_box()
        cx = (x0 + x1) / 2.0 / width
        cy = (y0 + y1) / 2.0 / height
        w = (x1 - x0) / width
        h = (y1 - y0) / height
        label = label_path_for(self.labels_dir, self.images[index])
        label.write_text(
            f"{self.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8"
        )
        print(f"saved {label.name}")

    def delete(self) -> None:
        label = label_path_for(self.labels_dir, self.images[self.index])
        if label.is_file():
            label.unlink()
            print(f"deleted {label.name}")
        self.box = None

    def _ordered_box(self) -> tuple[int, int, int, int]:
        assert self.box is not None
        x0, y0, x1, y1 = self.box
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    # --------------------------------------------------------------- callbacks
    def on_mouse(self, event, x, y, flags, param) -> None:
        cv2 = self.cv2
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.origin = (x, y)
            self.box = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.box = (self.origin[0], self.origin[1], x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.box = (self.origin[0], self.origin[1], x, y)
            # Echo to the terminal: if a drag produces no line here, the GUI is
            # not delivering mouse events at all (which is a different problem
            # from the box being drawn but invisible).
            x0, y0, x1, y1 = self._ordered_box()
            print(f"[label_cli] box ({x0},{y0})-({x1},{y1})  ->  press s to save")

    def run(self) -> int:
        cv2 = self.cv2
        import sys

        print(
            f"[label_cli] python : {sys.executable}\n"
            f"[label_cli] opencv : {cv2.__version__}  {cv2.__file__}\n"
            f"[label_cli] images : {len(self.images)} in {self.images[0].parent}\n"
            f"[label_cli] labels : {self.labels_dir}"
        )
        try:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            raise RuntimeError(
                "无法创建 OpenCV 窗口：当前解释器里的 cv2 不支持 GUI。\n"
                f"  cv2 路径: {cv2.__file__}\n"
                "  常见原因：装的是 opencv-python-headless（例如 ~/.local 下那个），"
                "它编译时没有 GUI 后端。\n"
                "  解决：用本工作区 venv 里的完整版运行，不要用系统 python3：\n"
                "    .venv-perception/bin/label_dataset --images <图片目录>"
            ) from exc
        # Realise the window with a first frame BEFORE binding the mouse
        # callback: some HighGUI backends only deliver mouse events once the
        # window has actually been shown.
        loaded_index = self.index
        image = self.load_image(self.index)
        self.box = self.load_box(self.index, image)
        cv2.imshow(self.window, image)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window, self.on_mouse)
        print(
            "[label_cli] 请在窗口里按住鼠标左键拖出一个框；"
            "拖完终端会打印框坐标，然后按 s 保存。"
        )

        while 0 <= self.index < len(self.images):
            if self.index != loaded_index:
                image = self.load_image(self.index)
                self.box = self.load_box(self.index, image)
                self.dragging = False
                loaded_index = self.index

            display = image.copy()
            if self.box is not None:
                x0, y0, x1, y1 = self._ordered_box()
                color = (0, 220, 0) if self.dragging else (60, 200, 255)
                cv2.rectangle(display, (x0, y0), (x1, y1), color, 3)
                if self.dragging:
                    cv2.putText(
                        display,
                        "release to finish",
                        (x0, max(y0 - 10, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )
            header = (
                f"{self.index + 1}/{len(self.images)}  {self.images[self.index].name}"
                f"  class={self.class_name}"
                f"  {'[box set - press s]' if self.box else '[drag a box]'}"
            )
            cv2.putText(
                display, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5
            )
            cv2.putText(
                display, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.imshow(self.window, display)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                self.save(self.index, image)
                self.index += 1
            elif key in (ord("n"), ord(" ")):
                self.index += 1
            elif key == ord("d"):
                self.delete()
            elif key == ord("b"):
                self.index = max(0, self.index - 1)
        cv2.destroyAllWindows()
        labeled = sum(
            1 for image in self.images if read_label_classes(label_path_for(self.labels_dir, image))
        )
        print(f"{labeled}/{len(self.images)} images labelled in {self.labels_dir}")
        return 0


def write_outlier_montage(
    report, labels_dir: Path, class_name: str, output: Path, max_tiles: int = 12
) -> None:
    """Render the most suspicious labelled frames, worst first."""
    from .viz import draw_label_boxes, montage

    tiles = []
    for item in report.flagged[:max_tiles]:
        frame = np.asarray(Image.open(item.image).convert("RGB"))
        tiles.append(
            draw_label_boxes(
                frame, label_path_for(labels_dir, item.image), class_name
            )
        )
    if not tiles:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(montage(tiles, columns=3)).save(output)
    print(f"[label_cli] 可疑样本拼图 -> {output}")


def write_preview(
    images: list[Path], labels_dir: Path, class_name: str, output: Path
) -> int:
    """Render existing labels over every image into one montage."""
    from .viz import draw_label_boxes, montage

    tiles = []
    labelled = 0
    for image_path in images:
        frame = np.asarray(Image.open(image_path).convert("RGB"))
        label = label_path_for(labels_dir, image_path)
        if read_label_classes(label):
            labelled += 1
        tiles.append(draw_label_boxes(frame, label, class_name))
    canvas = montage(tiles, columns=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(output)
    print(
        f"[label_cli] {labelled}/{len(images)} images labelled; "
        f"preview written to {output}"
    )
    return 0


def main(argv=None) -> int:
    from .launcher import ensure_runtime

    ensure_runtime("huanpan_perception.label_cli")

    args = parse_args(argv)
    try:
        images_dir = Path(args.images).expanduser()
        labels_dir = (
            Path(args.labels).expanduser()
            if args.labels
            else images_dir.parent / "labels"
        )
        if args.check:
            from .labelcheck import check_dataset, format_report

            images = list_images(images_dir)
            if not images:
                print(f"[label_cli] no images found in {args.images}")
                return 2
            report = check_dataset(images_dir, labels_dir)
            print(format_report(report))
            write_outlier_montage(
                report,
                labels_dir,
                args.class_name,
                Path(args.check_output).expanduser(),
            )
            return 0
        if args.preview:
            images = list_images(images_dir)
            if not images:
                print(f"[label_cli] no images found in {args.images}")
                return 2
            return write_preview(
                images, labels_dir, args.class_name, Path(args.preview).expanduser()
            )
        return Annotator(args).run()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[label_cli] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
