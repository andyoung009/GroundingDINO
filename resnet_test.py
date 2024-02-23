from transformers import AutoImageProcessor, ResNetModel
import torch
# from datasets import load_dataset
import cv2

# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]

image_folder_path = '/data/ML_document/datasets/custom_6dpose_dataset/train/19.png'
image = cv2.imread(image_folder_path)

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetModel.from_pretrained("microsoft/resnet-50")

inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)