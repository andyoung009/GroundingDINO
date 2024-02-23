import os
from PIL import Image
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import cv2
HOME = os.getcwd()
print(HOME)

CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))

WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))

# %cd {HOME}/GroundingDINO
code_dir = os.path.join(HOME, "GroundingDINO")
os.chdir(code_dir)

from groundingdino.util.inference import load_model, load_image, predict, annotate

model = load_model(CONFIG_PATH, WEIGHTS_PATH)

# IMAGE_NAME = "dog-3.jpeg"
# IMAGE_NAME_Grasp = "/data/ML_document/datasets/custom_6dpose_dataset/val/9.png"
IMAGE_NAME_Grasp = "/data/ML_document/datasets/custom_6dpose_dataset/val/33.png"
# IMAGE_PATH = os.path.join(HOME, "data", IMAGE_NAME)

# TEXT_PROMPT = "chair"
TEXT_PROMPT_Grasp = "use the robot end gripper, to grasp milk carton"

BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_NAME_Grasp)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT_Grasp,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
# opencv differ to other lib color space, change BGR2RGB
image_ann_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

image_ann = Image.fromarray(image_ann_rgb)
# # the right RGB mode
# image_ann.save('/data/ML_document/datasets/custom_6dpose_dataset/outputs/9.png')
# # the wrong BGR mode
# plt.imsave('/data/ML_document/datasets/custom_6dpose_dataset/outputs/9_1.png', annotated_frame)

# the right RGB mode
image_ann.save('/data/ML_document/datasets/custom_6dpose_dataset/outputs/33.png')
# the wrong BGR mode
plt.imsave('/data/ML_document/datasets/custom_6dpose_dataset/outputs/33_1.png', annotated_frame)