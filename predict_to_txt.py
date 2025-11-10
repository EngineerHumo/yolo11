"""Predict script for running YOLO detections on a directory of images and saving results in YOLO format."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_MODEL_PATH = "/home/wensheng/jiaqi/ultralytics/runs/detect/train39/weights/best.pt"
DEFAULT_SOURCE_DIR = "/home/wensheng/jiaqi/ultralytics/video"
OUTPUT_DIR = Path("/home/wensheng/jiaqi/ultralytics/output_txt")

BOTTOM_CROP_PIXELS = 38
TARGET_IMAGE_SIZE = 1240


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO detections on a directory of images and save YOLO format txt files."
    )
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


def run_inference(model: YOLO, image: np.ndarray):
    return model.predict(source=image, verbose=False, save=False)


def prepare_image(image: np.ndarray) -> np.ndarray:
    """Crop and resize the image to the 1240x1240 target required by the pipeline."""

    height, width = image.shape[:2]
    if height <= BOTTOM_CROP_PIXELS:
        raise ValueError(
            "Image height must be greater than the bottom crop value of"
            f" {BOTTOM_CROP_PIXELS}, got {height}."
        )

    cropped = image[: height - BOTTOM_CROP_PIXELS, :, :]

    if cropped.shape[0] != TARGET_IMAGE_SIZE or cropped.shape[1] != TARGET_IMAGE_SIZE:
        cropped = cv2.resize(
            cropped,
            (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

    return cropped


def save_detections_to_txt(output_path: Path, detections) -> None:
    txt_path = output_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as file:
        for result in detections:
            boxes = getattr(result.boxes, "xywhn", None)
            class_ids = getattr(result.boxes, "cls", None)
            if boxes is None or class_ids is None:
                continue
            boxes_np = boxes.cpu().numpy()
            class_ids_list = class_ids.int().cpu().tolist()
            for class_id, (x_center, y_center, width, height) in zip(class_ids_list, boxes_np):
                file.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                )


def process_images(model: YOLO, images: List[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image {image_path!s}")
        processed_image = prepare_image(image)
        detections = run_inference(model, processed_image)
        output_image_path = output_dir / image_path.name
        cv2.imwrite(str(output_image_path), processed_image)
        save_detections_to_txt(output_image_path, detections)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source)

    model = YOLO(args.model)

    images = collect_images(source_dir)
    process_images(model, images, OUTPUT_DIR)

    print(
        "Cropped 1240x1240 images and YOLO txt files saved to"
        f" {OUTPUT_DIR!s}."
    )


if __name__ == "__main__":
    main()
