from typing import Tuple

from mmdet.apis import inference_detector, init_detector
import mmengine
import numpy as np


class MMDetModel:
    def __init__(self, weights_path: str, cfg_path: str, device: str = 'cpu'):
        cfg = mmengine.Config.fromfile(cfg_path)
        
        cfg.model.test_cfg.rcnn.max_per_img = 1000
        cfg.model.roi_head.bbox_head.num_classes = 3
        cfg.model.test_cfg.rpn.nms_pre = 3000
        cfg.model.test_cfg.rpn.max_per_img = 2000
        cfg.test_cfg = None
        
        self.model = init_detector(cfg, weights_path, device=device)
        self.model.cfg = cfg

    def select(self, results: np.array, threshold: float) -> Tuple:
        boxes = results.pred_instances.bboxes.numpy()
        scores = results.pred_instances.scores.numpy()[:, np.newaxis]
        labels = results.pred_instances.labels.numpy()[:, np.newaxis] + 1
        mask = (scores > threshold).flatten()
        
        return boxes[mask], labels[mask], scores[mask]

    def predict(self, image: np.array, threshold: float = .0) -> Tuple:
        results = inference_detector(self.model, image)
        selected = self.select(results, threshold)
        return selected
