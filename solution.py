from typing import List
from pathlib import Path

import numpy as np
from PIL import Image

from models import MMDetModel

model = MMDetModel(
    weights_path='./weights/faster-rcnn-best.pth',
    cfg_path='./configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
)


def _glob_images(folder: Path, exts: List[str] = ('*.jpg', '*.png',)) -> List[Path]:
    images = []
    for ext in exts:
        images += list(folder.glob(ext))
    return images


def format_predictions(labels, scores, bboxes) -> str:
    formatted = []

    for label, score, bbox in zip(labels, scores, bboxes):
        label = int(label)
        score = round(float(score), 4)
        xmin, ymin, xmax, ymax = map(int, bbox)

        line = f"{label} {score} {xmin} {ymin} {xmax} {ymax}"
        formatted.append(line)

    return "\n".join(formatted)


def write_predictions(predictions: str, output_path: Path) -> None:
    with open(output_path, 'w') as f:
        f.write(predictions)


def predict_folder(input_folder: str, output_folder: str) -> None:
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    images_path = _glob_images(input_folder)

    for img_path in images_path:

        img = np.asarray(Image.open(img_path))
        bboxes, labels, scores = model.predict(img)
        predictions_repr = format_predictions(labels, scores, bboxes)
        output_path = output_folder / img_path.with_suffix('.txt').name

        write_predictions(predictions_repr, output_path)


def main():
    input_folder = './private/images'
    output_folder = './output'

    predict_folder(input_folder, output_folder)


if __name__ == '__main__':
    main()
        