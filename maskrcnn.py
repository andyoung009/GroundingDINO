# modified from https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=8IRGo8d0qkgR 
# by andyoung0507
# 利用maskrcnn输出的分类结果和掩码，输入到MLP中去输出动作，但是从测试结果来看并不理想，是否是采集的数据的精度不高呢？
# 首先预训练模型是coco数据集训练的，这样分类的结果是否可以与语言输入的特征匹配？
# 其次输出的精度不太好，图片效果可以看 outputs/15_maskrcnn.png 
# 代码输出的结果的标签是coco dataset 2017格式，具体对应关系可以参考这里 https://github.com/ultralytics/yolov5/blob/master/data/coco.yaml
# 39: bottle对应抓取的物体为盒子，但是不能识别 牛奶盒(clip利用语言和图像相似度可以进行预测但无法给出位置信息)：milk carton
import torch, detectron2

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

IMAGE_NAME_Grasp = "/data/ML_document/datasets/custom_6dpose_dataset/val/33.png"
im = cv2.imread(IMAGE_NAME_Grasp)
print(im.shape)

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite("/data/ML_document/datasets/custom_6dpose_dataset/outputs/33_maskrcnn.png", out.get_image()[:, :, ::-1])
# cv2.imwrite("/data/ML_document/datasets/custom_6dpose_dataset/outputs/33_maskr_save.png", outputs['instances'].pred_masks)
print("test!")

if 39 in outputs['instances'].pred_classes.tolist():
    # bottle in the outputs pred_classes list, get the bottle index and filter the message in outputs['instances']
    index = outputs['instances'].pred_classes.tolist().index(39)
    target_mask = outputs['instances'].pred_masks.tolist()[index]
    target_mask_tensor = torch.tensor(target_mask)
    target_boxes = outputs['instances'].pred_boxes.tolist()[index]
    target_boxes_tensor = torch.tensor(target_boxes)
    target_scores = outputs['instances'].scores.tolist()[index]
    target_scores_tensor = torch.tensor(target_scores)
else:
    pass