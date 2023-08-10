# 本程序用作主干模型网络的相关特征图信息的输出

import torch

# b = batch_shape = 12
# h = w =224
# dtype = torch.float32
# device = 'cuda'

# tensor_list = torch.randn(6,3,224,224)

# tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
# mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
# for img, pad_img, m in zip(tensor_list, tensor, mask):
#     pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
#     m[: img.shape[1], : img.shape[2]] = False



import os
HOME = os.getcwd()
print(HOME)

import os
from groundingdino.util.inference import load_model, load_image, predict, annotate
CONFIG_PATH = os.path.join(HOME, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))

# 企图实现Grounding DINO的模型计算图可视化，尝试未果，分别试了tensorboard torchviz networkx netron 
# from torchsummary import summary
import os
from torchviz import make_dot
from IPython.display import display
import pydot

import networkx as nx
import matplotlib.pyplot as plt

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

CONFIG_PATH = os.path.join(HOME, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))

WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))

IMAGE_NAME = "dog-3.jpeg"
IMAGE_PATH = os.path.join(HOME, "data", IMAGE_NAME)

TEXT_PROMPT = "chair"
TEXT_PROMPT = preprocess_caption(TEXT_PROMPT)
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

model = load_model(CONFIG_PATH, WEIGHTS_PATH)

text_exaple = torch.randn(1)

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model, 
    image=image, 
    caption=TEXT_PROMPT, 
    box_threshold=BOX_TRESHOLD, 
    text_threshold=TEXT_TRESHOLD
)