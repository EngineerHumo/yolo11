"""Predict script for running YOLO detections on a directory of images and saving results in YOLO format."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

DEFAULT_MODEL_PATH = "/home/wensheng/jiaqi/ultralytics/runs/detect/train13/weights/best.pt"
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


def collect_images(source_dir: Path) -> list[Path]:
    images: list[Path] = sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() == ".jpg"
    )
    if not images:
        raise FileNotFoundError(f"No JPG images found in {source_dir!s}")
    return images


def run_inference(model: YOLO, image: np.ndarray):
    return model.predict(source=image, verbose=False, save=False, prob=True)


def prepare_image(image: np.ndarray) -> np.ndarray:
    """Crop and resize the image to the 1240x1240 target required by the pipeline."""
    height, _width = image.shape[:2]
    if height <= BOTTOM_CROP_PIXELS:
        raise ValueError(
            f"Image height must be greater than the bottom crop value of {BOTTOM_CROP_PIXELS}, got {height}."
        )

    cropped = image[: height - BOTTOM_CROP_PIXELS, :, :]

    if cropped.shape[0] != TARGET_IMAGE_SIZE or cropped.shape[1] != TARGET_IMAGE_SIZE:
        cropped = cv2.resize(
            cropped,
            (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

    return cropped


NUM_CLASSES = 6


def save_detections_to_txt(output_path: Path, detections) -> None:
    txt_path = output_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as file:
        for result in detections:
            boxes = getattr(result.boxes, "xywhn", None)
            class_ids = getattr(result.boxes, "cls", None)
            confidences = getattr(result.boxes, "conf", None)
            probs = getattr(result, "probs", None)

            if boxes is None:
                continue
            boxes_np = boxes.cpu().numpy()
            class_ids_list = class_ids.int().cpu().tolist() if class_ids is not None else [None] * len(boxes_np)

            probs_np = None
            if probs is not None:
                probs_tensor = getattr(probs, "data", probs)
                if probs_tensor is not None:
                    probs_np = probs_tensor.cpu().numpy()

            if probs_np is not None and len(probs_np) == len(boxes_np):
                confidences_np = probs_np[:, :NUM_CLASSES]
            else:
                confidences_np = np.zeros((len(boxes_np), NUM_CLASSES), dtype=float)
                if confidences is not None:
                    conf_list = confidences.cpu().numpy().tolist()
                else:
                    conf_list = [0.0] * len(boxes_np)
                for idx, (class_id, conf_value) in enumerate(zip(class_ids_list, conf_list)):
                    if class_id is not None and 0 <= class_id < NUM_CLASSES:
                        confidences_np[idx][int(class_id)] = float(conf_value)

            for confidence_row, (x_center, y_center, width, height) in zip(confidences_np, boxes_np):
                confidence_str = " ".join(f"{conf_value:.6f}" for conf_value in confidence_row)
                file.write(f"{confidence_str} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def process_images(model: YOLO, images: list[Path], output_dir: Path) -> None:
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

    print(f"Cropped 1240x1240 images and YOLO txt files saved to {OUTPUT_DIR!s}.")


if __name__ == "__main__":
    main()
