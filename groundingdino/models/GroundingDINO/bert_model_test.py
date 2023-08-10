# 本程序用作Bert网络的相关特征图信息的输出

import os
import torch
# import time
os.environ["HF_HOME"] = "/data/ML_document/.cache/huggingface"
from transformers import BertTokenizer, BertModel, AutoTokenizer
from bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)

text_encoder_type = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 模仿Groundingdino中Bert部分的流程，为了输出中间隐藏层特征信息，设置output_hidden_states=True
model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
model.pooler.dense.weight.requires_grad_(False)
model.pooler.dense.bias.requires_grad_(False)
model = BertModelWarper(bert_model=model)
print(model.config.hidden_size)
text = 255 * "test."
encoded_input = tokenizer(text, add_special_tokens=True, return_tensors='pt')
# print(encoded_input)
with torch.no_grad():
    output = model(**encoded_input)
layer_features = output.hidden_states
print(type(output))
print(output['last_hidden_state'].shape)
print(output['pooler_output'].shape)

# Access the layer-wise features
for i, layer in enumerate(layer_features):
    print(f"Layer {i}: {layer.shape}")