from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

model = YOLO(model='weights/yolov5nu.pt', task='detect')

def convert_corners_to_center_relative(x1, y1, x2, y2, img_width, img_height):
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1

    # Convert coordinates and dimensions to be relative to the image size
    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height

    return [center_x, center_y, width, height]

def _glob_images(folder: Path, exts: List[str] = ['*.jpg', '*.png',]) -> List[Path]:
    images = []
    for ext in exts:
        images += list(folder.glob(ext))
    return images

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.width, img.height

def format_predictions_to_str(img_path, preds) -> List[str]:
    img_width, img_height = get_image_dimensions(img_path)

    xyxy = preds.boxes.xyxy.cpu().numpy()
    labels = preds.boxes.cls.cpu().numpy()

    formatted_preds = []
    for idx, bbox in enumerate(xyxy):
        center_x, center_y, width, height = convert_corners_to_center_relative(bbox[0], bbox[1], bbox[2], bbox[3], img_width, img_height)
        line = f"{int(labels[idx])} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
        formatted_preds.append(line)
    return formatted_preds

def predict_folder(input_folder: str, output_folder: str):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)  # Create the output directory if it doesn't exist

    images_path = _glob_images(input_folder)

    for img_path in images_path:
        preds = model(img_path)[0]
        predictions_lines = format_predictions_to_str(img_path, preds)

        # Constructing the output path
        output_path = output_folder / f"{img_path.stem}.txt"

        # Writing to the output file
        with open(output_path, 'w') as f:
            for line in predictions_lines:
                f.write(line + '\n')

def main():
    input_folder = './test_data/images'
    output_folder = './output'

    predict_folder(input_folder, output_folder)

if __name__ == '__main__':
    main()
