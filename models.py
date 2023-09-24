from typing import Tuple

import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2

class MMDetModel:
    def __init__(self, weights_path: str, cfg_path: str, device: str = 'cpu'):
        
        self.model = AutoDetectionModel.from_pretrained(
            model_type='detectron2',
            model_path=weights_path,
            config_path=cfg_path,
            image_size=640,
            confidence_threshold=0.5,
            device=device
        )

    def predict(self, image, threshold: float = .0) -> Tuple:
        contrast_factor = 1.1
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        image = cv2.filter2D(image, -1, kernel)
        image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)

        results = get_sliced_prediction(
            image,
            self.model,
            slice_height = 256,
            slice_width = 256,
            overlap_height_ratio = 0.2,
            overlap_width_ratio = 0.2
        )

        return results.to_coco_predictions()
