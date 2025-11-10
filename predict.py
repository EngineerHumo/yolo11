"""Predict script for running YOLO detections on a directory of images.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
from ultralytics import YOLO


DEFAULT_MODEL_PATH = "/home/wensheng/jiaqi/ultralytics/runs/detect/train53/weights/best.pt"
DEFAULT_SOURCE_DIR = "C:/Users/jiaqi.guo/testproject/video"
DEFAULT_OUTPUT_SUBDIR = "predictions"
DEFAULT_VIDEO_NAME = "predictions.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detections on a directory of images.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained YOLO model weights.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing JPG images for detection.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Name of the subdirectory where annotated images will be saved.",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default=DEFAULT_VIDEO_NAME,
        help="Name of the output MP4 video file created from annotated images.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Frames per second for the output video.",
    )
    return parser.parse_args()


def collect_images(source_dir: Path) -> List[Path]:
    images: List[Path] = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".jpg"
    )
    if not images:
        raise FileNotFoundError(f"No JPG images found in {source_dir!s}")
    return images


def ensure_output_dir(source_dir: Path, output_name: str) -> Path:
    output_dir = source_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_annotated_images(model: YOLO, images: List[Path], output_dir: Path) -> List[Path]:
    saved_images: List[Path] = []
    for image_path in images:
        results = model(image_path)
        for result in results:
            annotated = result.plot()  # Different classes receive unique colors by default.
            output_path = output_dir / image_path.name
            if not cv2.imwrite(str(output_path), annotated):
                raise RuntimeError(f"Failed to write annotated image to {output_path!s}")
            saved_images.append(output_path)
    return saved_images


def build_video_from_images(images: List[Path], output_path: Path, fps: float) -> None:
    first_image = cv2.imread(str(images[0]))
    if first_image is None:
        raise RuntimeError(f"Unable to read the first annotated image {images[0]!s}")

    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Unable to read annotated image {image_path!s}")
            video_writer.write(frame)
    finally:
        video_writer.release()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source)
    output_dir = ensure_output_dir(source_dir, args.output_name)

    model = YOLO(args.model)

    images = collect_images(source_dir)
    saved_images = save_annotated_images(model, images, output_dir)

    video_path = output_dir / args.video_name
    build_video_from_images(saved_images, video_path, args.fps)

    print(f"Annotated images saved to: {output_dir!s}")
    print(f"Video saved to: {video_path!s}")


if __name__ == "__main__":
    main()
