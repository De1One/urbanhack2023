import sys, os, distutils.core
import torch, detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
from matplotlib import pyplot as plt
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data.datasets import register_coco_instances
import detectron2.data.transforms as T
from detectron2.model_zoo import get_config
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import copy
from contextlib import redirect_stdout
setup_logger()

#train and validation datasets
register_coco_instances("my_dataset_train1", {}, "/kaggle/input/trainhack/urbanhack-train/urbanhack-train/annotations/instances_default.json",
                        "/kaggle/input/trainhack/urbanhack-train/urbanhack-train/images")
register_coco_instances("my_dataset_val1", {}, "/kaggle/input/trainhack/urbanhack-train/urbanhack-train/annotations/instances_test.json",
                        "/kaggle/input/trainhack/urbanhack-train/urbanhack-train/images")

def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [
        T.Resize([960,640]),
        T.RandomBrightness(0.8, 1.3),
        T.RandomContrast(0.8, 1.2),
        T.RandomSaturation(0.8, 1.3),
        T.RandomRotation(angle=[25, 40]),
        T.RandomLighting(0.9),
        T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train1",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00015
cfg.SOLVER.MAX_ITER = 400
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
#cfg.MODEL.RETINANET.NUM_CLASSES = 3


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.SOLVER.BASE_LR = 0.0003
cfg.SOLVER.MAX_ITER = 2000
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)


evaluator = COCOEvaluator("my_dataset_val1", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "my_dataset_val1")
print(inference_on_dataset(predictor.model, val_loader, evaluator))


im = cv2.imread('/kaggle/input/trainhack/urbanhack-train/urbanhack-test/1411_14_36_12-2023-09-11.jpg')
outputs = predictor(im)
v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
plt.imshow(out.get_image()[:, :, ::-1])
plt.show()


with open('test.yml', 'w') as f:
    with redirect_stdout(f): print(cfg.dump())