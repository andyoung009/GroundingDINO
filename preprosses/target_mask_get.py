# modified from https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=8IRGo8d0qkgR 
# by andyoung0507
# 利用maskrcnn输出的分类结果和掩码，输入到MLP中去输出动作，但是从测试结果来看并不理想，
# 首先预训练模型是coco数据集训练的，这样分类的结果是否可以与语言输入的特征匹配？
# 其次输出的精度不太好，图片效果可以看 outputs/15_maskrcnn.png 
# 代码输出的结果的标签是coco dataset 2017格式，具体对应关系可以参考这里 https://github.com/ultralytics/yolov5/blob/master/data/coco.yaml
# 39: bottle; 41：cup 对应抓取的物体为瓶子, 但是不能识别 牛奶盒：milk carton
import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

# Define the folder paths
# train_image_folder
image_folder = '/data/ML_document/datasets/custom_6dpose_dataset/train/'
# val_image_folder
# image_folder = '/data/ML_document/datasets/custom_6dpose_dataset/val/'
output_folder = '/data/ML_document/datasets/custom_6dpose_dataset/outputs_target_mask/'

# Load the Mask R-CNN model
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# model = torch.hub.load('facebookresearch/detectron2:main', 'mask_rcnn_R_50_FPN_3x')

# Set the model to evaluation mode
# model.eval()

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over the PNG images in the input folder
for filename in os.listdir(image_folder):
    if filename.endswith('.png'):
        # Load and preprocess the image
        image_path = os.path.join(image_folder, filename)
        im = cv2.imread(image_path)
        # image = Image.open(image_path)
        # preprocess = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        # input_tensor = preprocess(image)
        # input_batch = input_tensor.unsqueeze(0)

        # # Run inference
        # with torch.no_grad():
        #     output = model(input_batch)
        outputs = predictor(im)

        # Extract class labels, scores, and masks
        labels = outputs['instances'].pred_classes.tolist()

        if 39 or 41 in labels:
            if 39 in labels:
                class_label = 39
                class_name = 'bottle'
            elif 41 in labels:
                class_label = 41
                class_name = 'cup'
            else:
                continue
            index = outputs['instances'].pred_classes.tolist().index(class_label)
            # target_mask = outputs['instances'].pred_masks.tolist()[index]
            target_mask = outputs['instances'].pred_masks[index].cpu().numpy()
            # target_mask_tensor = torch.tensor(target_mask)
            target_boxes = outputs['instances'].pred_boxes[index].tensor.cpu()
            target_boxes_tensor = torch.tensor(target_boxes)
            target_scores = outputs['instances'].scores.tolist()[index]
            target_scores_tensor = torch.tensor(target_scores)

        # scores = outputs['instances'].scores.tolist()
        # masks = outputs['instances'].pred_masks.cpu().numpy()

        # Save the results as a text file
            output_filename = os.path.splitext(filename)[0] + '.txt'
            output_filepath = os.path.join(output_folder, output_filename)
            with open(output_filepath, 'w') as f:


                f.write(f"Label: {class_name}, Score: {target_scores_tensor}\n")
                f.write(f"Box of : {class_name}, Box: {target_boxes}\n")
                f.write(f"Mask: {target_mask}\n\n")
            target_mask_filename = os.path.splitext(filename)[0] + '.npy'
            target_mask_filepath = os.path.join(output_folder, target_mask_filename)
            np.save(target_mask_filepath, target_mask)

        # Save the predicted masks as image files
        # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imwrite("/data/ML_document/datasets/custom_6dpose_dataset/outputs/33_maskrcnn.png", out.get_image()[:, :, ::-1])


            mask_image = Image.fromarray((target_mask * 255).astype('uint8'), mode='L')
            mask_filename = os.path.splitext(filename)[0] + f"_mask_target.png"
            mask_filepath = os.path.join(output_folder, mask_filename)
            mask_image.save(mask_filepath)

        else:
            pass