import os
import numpy as np
from pathlib import Path

def center_to_corners(xc, yc, w, h):
    """Convert center-width-height format to corner coordinates."""
    x1 = xc - w/2
    y1 = yc - h/2
    x2 = xc + w/2
    y2 = yc + h/2
    return x1, y1, x2, y2


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes."""
    x1c, y1c, w1, h1 = box1
    x2c, y2c, w2, h2 = box2

    x1, y1, x2, y2 = center_to_corners(x1c, y1c, w1, h1)
    x1g, y1g, x2g, y2g = center_to_corners(x2c, y2c, w2, h2)

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    
    return iou

def compute_ap(rec, prec):
    """Compute average precision."""
    mrec = [0.] + [e for e in rec] + [1.]
    mpre = [0.] + [e for e in prec] + [0.]
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = [i for i in range(len(mrec) - 1) if mrec[i] != mrec[i + 1]]
    ap = sum([(mrec[i + 1] - mrec[i]) * mpre[i + 1] for i in idx])
    return ap

def compute_map_50(ground_truth_dir, prediction_dir):
    gt_dir = Path(ground_truth_dir)
    pred_dir = Path(prediction_dir)

    all_files = list(gt_dir.glob("*.txt"))
    
    total_gt = {}
    total_preds = {}
    all_labels = set()

    # Count total ground truths per label
    for file in all_files:
        with file.open('r') as f:
            for line in f:
                label, *bbox = line.strip().split()
                all_labels.add(label)
                total_gt[label] = total_gt.get(label, 0) + 1

    # Initialize total predictions to zero
    for label in all_labels:
        total_preds[label] = 0

    scores = {label: [] for label in all_labels}
    matched_detections = {label: 0 for label in all_labels}

    # Match preds to ground truth
    for file in all_files:
        gt_bboxes = {}
        pred_bboxes = {}
        for label in all_labels:
            gt_bboxes[label] = []
            pred_bboxes[label] = []

        # Load ground truth
        with file.open('r') as f:
            for line in f:
                label, *bbox = line.strip().split()
                bbox = [float(x) for x in bbox]
                gt_bboxes[label].append(bbox)

        pred_file = pred_dir / file.name
        if pred_file.exists():
            with pred_file.open('r') as f:
                for line in f:
                    label, *bbox = line.strip().split()
                    if label in all_labels:
                        bbox = [float(x) for x in bbox]
                        pred_bboxes[label].append(bbox)
                        total_preds[label] += 1

        for label in all_labels:
            detected = []
            for pred in pred_bboxes[label]:
                # print(f"Pred Box: {pred}")  # Debugging line
                best_iou = 0.5
                best_gt_idx = -1
                for idx, gt in enumerate(gt_bboxes[label]):
                    # print(f"GT Box: {gt}")  # Debugging line
                    if idx not in detected:
                        iou = compute_iou(pred, gt)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx

                if best_gt_idx >= 0:
                    detected.append(best_gt_idx)
                    matched_detections[label] += 1
                scores[label].append((best_iou, 1 if best_gt_idx >= 0 else 0))

    mAP = 0
    for label in all_labels:
        scores[label].sort(key=lambda x: x[0], reverse=True)
        tp_cumsum = np.cumsum([score[1] for score in scores[label]])
        precisions = tp_cumsum / (1 + np.arange(len(scores[label])))
        recalls = tp_cumsum / total_gt[label]
        mAP += compute_ap(recalls, precisions)
    
    mAP /= len(all_labels)
    return mAP

ground_truth_dir = "./test_data/labels"
prediction_dir = "./output/"
print(compute_map_50(ground_truth_dir, prediction_dir))
