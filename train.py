# --------------------------------------------------------
# train code modefied from yunjey/pytorch-tutorial
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py#L35-L56
# By yhx
# --------------------------------------------------------

import torch 
import torch.nn as nn
import torchvision
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from groundingdino.util.inference import load_model, load_image, predict, annotate, preprocess_caption
from rgbd_vla import RGBD2pose
from torch.utils.data import DataLoader
from data.custom_6dpose_dataset import IN2POSEDATASET
import random
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# https://pytorch.org/docs/stable/notes/randomness.html copy from here
# DataLoader will reseed workers following Randomness in multi-process data loading algorithm.
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(0)

# device = 'cpu'

# def custom_collate_fn(batch):
#     images = [torch.from_numpy(image) for image in batch]  # 将PIL.Image.Image转换为张量
#     return torch.stack(images)
    # return collate([torch.from_numpy(b) for b in batch])
def custom_collate_fn(batch):
    # 将numpy.uint16类型的数据转换为支持的数据类型
    batch[2] = [torch.tensor(data[2].astype(np.float32)) for data in batch]
    return batch

def random_seed(seed=1):
    random.seed(seed)


# Hyper parameters
num_epochs = 300
batch_size = 8 # 100
learning_rate = 0.0015

# data = pd.read_csv('/data/ML_document/datasets/custom_6dpose_dataset/custom_6dpose_dataset.csv')

# # 分割数据集
# train_dataset = data[data['split'] == 'train']
# val_dataset = data[data['split'] == 'val']

data = IN2POSEDATASET()
# train_dataset, val_dataset = torch.utils.data.random_split(data, [48, 13])
# use maskrcnn the dataset sample number becomes small
train_dataset, val_dataset = torch.utils.data.random_split(data, [37, 9])

# MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                            train=True, 
#                                            transform=transforms.ToTensor(),
#                                            download=True)

# test_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                           train=False, 
#                                           transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           worker_init_fn=seed_worker,
                                           generator=g,
                                           shuffle=True)
                                        #    collate_fn=custom_collate_fn)

test_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=batch_size, 
                                          worker_init_fn=seed_worker,
                                          generator=g,
                                          shuffle=False)
                                        #   collate_fn=custom_collate_fn)

model = RGBD2pose(device=device, lr=1e-3, hidden_dim=520)

# Loss and optimizer
# update_params = [model.e2pose, model.dep_fea_extra.depth_feature_extract.fc]
update_params = []
for param in model.e2pose.parameters():
    update_params.append(param)
# update_params += list(model.dep_fea_extra.depth_feature_extract)

criterion = nn.CrossEntropyLoss()
criterion_1 = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(update_params, lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, data_detailed in enumerate(train_loader):
        # (instructions, images_rgb, images_depth, outputs)
        instructions, images_rgb, images_depth, mask, outputs = data_detailed
        images_rgb = images_rgb.to(device)
        images_depth = images_depth.to(device)
        labels = outputs.to(device)
        mask = mask.to(device)
        # labels = torch.tensor(labels).long()
        
        # Forward pass
        outputs = model(images_rgb, images_depth, instructions, mask)
        loss_loc = criterion_1(outputs[:3], labels[:3])
        loss_pos = criterion_1(outputs[-4:], labels[-4:])
        loss = 0.5 * loss_loc + 0.5 * loss_pos
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if True or (i+1) % 2 == 0:
            with open('./loss_log_Maskrcnn.txt', 'a') as f:
                f.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Loss_loc: {:.4f}, Loss_pos: {:.4f}\n' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item(), loss_loc.item(), loss_pos.item()))
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Loss_loc: {:.4f}, Loss_pos: {:.4f}\n' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item(), loss_loc.item(), loss_pos.item()))

    # Test the model
    if epoch % 10 == 0:
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            error = 0
            for instructions, images_rgb, images_depth, mask, outputs in test_loader:
                images_rgb = images_rgb.to(device)
                images_depth = images_depth.to(device)
                labels = outputs.to(device)
                mask = mask.to(device)
                
                outputs = model(images_rgb, images_depth, instructions, mask)
                loss_loc = criterion_1(outputs[:3], labels[:3])
                loss_pos = criterion_1(outputs[-4:], labels[-4:])
                loss = 0.5 * loss_loc + 0.5 * loss_pos
                
                total += labels.size(0)
                error += loss
            
            with open('./loss_log_Maskrcnn.txt', 'a') as f:
                f.write('The average error of the model on the 9 test images  is: {} '.format(error / total))

            with open('./loss_log_Maskrcnn.txt', 'a') as f:
                f.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, loss_loc: {:.4f}, loss_pos: {:.4f}\n' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item(), loss_loc.item(), loss_pos.item()))

            print('The average error of the model on the 9 test images  is: {}.'.format(error / total))
f.close()
# Save the model checkpoint
torch.save(model.state_dict(), './model.ckpt')