import torch
import torchvision.models as models
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.linear import Identity
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from groundingdino.util.inference import load_model, load_image, predict, annotate, preprocess_caption

# 配置文件、权重文件路径加载
# ------------------------------------------------------------------------------------
HOME = os.getcwd()
print(HOME)

CONFIG_PATH = os.path.join(HOME, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))

WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))
# --------------------------------------------------------------------------------------

# 深度图像特征提取网络
class Depth_Feature_extract(nn.Module):
    def __init__(self, device='cuda', lr=1e-3, output_dim=2048+4*8):
        super().__init__()
        self.depth_feature_extract = models.resnet50(pretrained=True)        
        self.depth_feature_extract.fc = nn.Identity()

        # self.depth_feature_extract.fc = nn.Linear(2048,output_dim)
        # nn.init.xavier_uniform_(self.depth_feature_extract.fc.weight)
        # nn.init.zeros_(self.depth_feature_extract.fc.bias)

    def forward(self, x):
        return self.depth_feature_extract(x)

class Embedding2pose(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class RGBD2pose(nn.Module):
    def __init__(self, device='cuda', lr=1e-3, hidden_dim=520):
        super().__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.backbone = load_model(CONFIG_PATH, WEIGHTS_PATH).to(device)
        
        # 替换GroundingDINO的transformer的末端，为了和后面微调网络连接，或者需要明确输出的格式，直接和e2pose相连接
        # self.transformer.decoder.bbox_embed = self.bbox_embed
        # self.transformer.decoder.class_embed = self.class_embed
        # self.backbone.transformer.decoder.bbox_embed = nn.ModuleList(Identity() for i in range(6))
        # self.backbone.transformer.decoder.class_embed = nn.ModuleList(Identity() for i in range(6))

        # 生成6d pose的网络架构（5层全连接网络）
        # self.e2pose = Embedding2pose(input_dim=4160,hidden_dim=hidden_dim,output_dim=7,num_layers=2)
        # 236080 = 900*(256+4) + 2080           (256+4)*8 = 2080
        # self.e2pose = Embedding2pose(input_dim=236080,hidden_dim=hidden_dim,output_dim=7,num_layers=2).to(device)
        self.e2pose = Embedding2pose(input_dim=4096,hidden_dim=hidden_dim,output_dim=7,num_layers=2).to(device)
        self.dep_fea_extra = Depth_Feature_extract(device=device,lr=1e-3, output_dim=2048+4*8).to(device)
        # self.e2pose.apply(initialize_weights)

    def preprocess_caption_instructions(self, instructions):
        processed = []
        for i in range(len(instructions)):
            processed.append(preprocess_caption(instructions[i]))
        return processed


    def forward(self, img_rgb, img_depth, instructions, mask):
        instructions = self.preprocess_caption_instructions(instructions)
        # 图片在前向推理时需要进行预处理，这部分代码先封装成为一个单独的函数
        img_rgb_out = self.backbone(img_rgb,instructions=instructions)

        # prediction_logits = img_rgb_out["pred_logits"].sigmoid()  # prediction_logits.shape = (B, nq, 256)
        # prediction_boxes = img_rgb_out["pred_boxes"] # prediction_boxes.shape = (nq, 4)
        # value_top8, indices_top8 = torch.topk(prediction_logits,8,dim=1)
        # remaintop8_pre_box = torch.gather(img_rgb_out["pred_boxes"], 1, indices_top8.unsqueeze(-1).repeat(1,1,1,4))


        # value_top8, indices_top8 = torch.topk(img_rgb_out["pred_logits"],8,dim=1)
        # remaintop8_pre_box = torch.gather(img_rgb_out["pred_boxes"], 1, indices_top8.unsqueeze(-1).repeat(1,1,1,4))


        # 图片深度信息复制为3通道同时将深度值变为原来的1/3
        # 引入mask，将mask和深度图像求均值作为第三通道，附加mask和深度图像通道最后输入到resnet50中
        mask = mask.float()
        depth_and_mask = (img_depth / 255.0 + mask) / 2
        total_depth_and_mask = torch.stack((mask, img_depth, depth_and_mask), dim=1)
        # img_depth = img_depth.unsqueeze(1).repeat(1,3,1,1) / 3.0 / 255.0
        # 特征提取拼接
        # img_depth_out = self.backbone(img_depth, instructions=instructions)
        img_depth_out = self.dep_fea_extra(total_depth_and_mask)

        # img_rgb_embed = torch.cat((img_rgb_out["pred_logits"], img_rgb_out["pred_boxes"]),dim=-1)
        B, N, C = img_rgb_out[0].shape
        deformed_img_rgb_out_0 = F.pad(img_rgb_out[0], (0, 0, 2,2), "constant", 0).reshape(B,-1,256*8)
        deformed_img_rgb_out_1 = torch.mean(deformed_img_rgb_out_0, dim=1)
        img_rgb_embed = torch.cat((deformed_img_rgb_out_1, img_depth_out),dim=-1)
        # img_depth_embed = torch.cat((img_depth_out["pred_logits"], img_depth_out["pred_boxes"]),dim=-1)
        # total_embed = torch.cat((img_rgb_embed,img_depth_embed),dim=-1) # total_embed.shape=(B,nq,(256+4)*2)

        # 提取Grounding DINO输出的特征的前top8与深度特征组合输入e2pose网络
        # B, remain_nq, C = value_top8.shape
        # value_top8_reshape = value_top8.reshape(B,-1)
        # remaintop8_pre_box_reshape = remaintop8_pre_box.reshape(B,-1)

        # B, remain_nq, C = img_rgb_out['pred_logits'].shape
        # img_rgb_out['pred_logits'] = img_rgb_out['pred_logits'].reshape(B,-1)
        # img_rgb_out['pred_boxes'] = img_rgb_out['pred_boxes'].reshape(B,-1)
        # total_embed = torch.cat((img_rgb_out['pred_logits'], img_rgb_out['pred_boxes'], img_depth_out),dim=-1)
        # pose_info = self.e2pose(total_embed)
        pose_info = self.e2pose(img_rgb_embed)
        return pose_info

