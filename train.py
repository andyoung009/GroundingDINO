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
import datetime
from scipy.spatial.transform import Rotation
from torch.nn.functional import normalize
import matplotlib.pyplot as plt

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

# 四元数表示姿态误差时采用的损失函数定义为quaternion_error函数，不直接利用四元数差异平方和作为loss_pos
def quaternion_to_euler(translation,rotation):
    loc = np.array([translation[0], translation[1], translation[2]])
    rot = Rotation.from_quat([rotation[0], rotation[1], rotation[2], rotation[3]]).as_matrix()
    RT = np.column_stack((rot,loc))
    RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
    return RT

# 自己写的，用起来有歧义不好用，调用上一个函数
def quaternion_to_rotation_matrix(quaternion):
    w, x, y, z = quaternion
    rotation_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x**2 + z**2), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x**2 + y**2)]
    ])
    return rotation_matrix

# def quaternion_error(true_quaternion, estimated_quaternion):
#     # 计算内积
#     inner_product = torch.matmul(true_quaternion, estimated_quaternion)
    
#     # 计算角度误差
#     error_angle = torch.acos(2 * (inner_product**2) - 1)
    
#     # 将弧度转换为角度
#     error_angle_degrees = torch.degrees(error_angle)
    
#     return error_angle_degrees

def quaternion_error(true_quaternions, estimated_quaternions):
    # 将四元数扩展一个维度，使得形状变为 (batch_size, 1, 4)

    r_t = Rotation.from_quat(true_quaternions.detach().cpu())
    euler_angles_t = r_t.as_euler('xyz', degrees=False)

    r_e = Rotation.from_quat(estimated_quaternions.detach().cpu())
    euler_angles_e = r_e.as_euler('xyz', degrees=False)

    error = euler_angles_t - euler_angles_e
    # 计算每个欧拉角的均方差
    mse = np.mean(np.square(error), axis=0)
    loss_pos = np.sum(mse)
    return loss_pos
    
    # true_quaternions = normalize(true_quaternions, p=2, dim=-1)
    # estimated_quaternions = normalize(estimated_quaternions, p=2, dim=-1)
    # true_quaternions = true_quaternions.unsqueeze(1)
    # estimated_quaternions = estimated_quaternions.unsqueeze(2)

    # # 计算内积，使用 torch.bmm 进行批量矩阵相乘
    # inner_product = torch.bmm(true_quaternions, estimated_quaternions).squeeze(-1)

    # # 将内积限制在 [-1, 1] 范围内
    # inner_product = torch.clamp(inner_product, -1.0, 1.0)

    # # 计算角度误差
    # error_angle = torch.acos(2 * (inner_product**2) - 1)

    # # 将弧度转换为角度
    # error_angle_degrees = torch.rad2deg(error_angle)

    # return error_angle_degrees

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
warmup_lr = 0.001
warmup_epochs = 5
num_epochs = 130
batch_size = 8 # 100
learning_rate = 0.01

data = pd.read_csv('/data/ML_document/datasets/custom_6dpose_dataset/custom_6dpose_dataset.csv')

# # 分割数据集
# train_dataset = data[data['split'] == 'train']
# val_dataset = data[data['split'] == 'val']

# 数据增强
transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),  # 随机水平翻转
    # transforms.RandomRotation(10),  # 随机旋转（-10到10度之间）
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机调整亮度、对比度、饱和度和色调
    # transforms.RandomResizedCrop(224),  # 随机裁剪和缩放到指定大小
    transforms.ToTensor(),  # 转换为张量
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# data = IN2POSEDATASET(transform=transform)
# 考虑到很多次训练精度无法提升可能是因为训练、验证数据集随机分割导致的，因此分开设置csv文件，分别读取train和val数据集
train_dataset = IN2POSEDATASET(ann_file='custom_6dpose_dataset_mask_train.csv', transform=transform)
val_dataset = IN2POSEDATASET(ann_file='custom_6dpose_dataset_mask_val.csv', transform=transform)
# train_dataset, val_dataset = torch.utils.data.random_split(data, [48, 13])
# use maskrcnn the dataset sample number becomes small
# train_dataset, val_dataset = torch.utils.data.random_split(data, [37, 9])

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
model.train()

# Loss and optimizer
# update_params = [model.e2pose, model.dep_fea_extra.depth_feature_extract.fc]
update_params = []
for param in model.e2pose.parameters():
    update_params.append(param)
# update_params.append()
update_params += list(model.dep_fea_extra.depth_feature_extract.conv1.parameters())

criterion = nn.CrossEntropyLoss()
criterion_1 = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer_wp = torch.optim.Adam(update_params, lr=warmup_lr)
optimizer = torch.optim.Adam(update_params, lr=learning_rate)
# optimizer_conv1 = torch.optim.Adam(model.conv1.parameters(), lr=0.001)  # 优化conv1层的参数
# optimizer_other = optim.SGD(model.fc.parameters(), lr=0.01)  # 优化其他层的参数

# 预热模型
print('Start the warm up!')
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
text_content = f"训练开始，当前时间是系统时间 {current_time}\n"

# 文件路径，可以根据实际情况修改
file_path = "./loss_log_Maskrcnn.txt"

# 将文本写入文件
with open('./loss_log_Maskrcnn.txt', 'a') as f:
    f.write(text_content)

print(f"训练开始日志已写入到文件: {file_path}")

for epoch in range(warmup_epochs):
    # 在预热阶段使用较小的学习率
    # 训练模型
    model.train()
    for i, data_detailed in enumerate(train_loader):
        instructions, images_rgb, images_depth, mask, outputs = data_detailed
        images_rgb = images_rgb.to(device)
        images_depth = images_depth.to(device)
        labels = outputs.to(device)
        mask = mask.to(device)
        outputs = model(images_rgb, images_depth, instructions, mask)
        loss_loc = criterion_1(outputs[:,:3], labels[:,:3])
        # loss_pos = criterion_1(outputs[:,-4:], labels[:,-4:])
        loss_pos = quaternion_error(outputs[:,-4:], labels[:,-4:])
        loss_pos = torch.from_numpy(np.array([loss_pos]))
        loss_pos = loss_pos.to(device)
        loss = 0.5 * loss_loc + 0.5 * torch.mean(loss_pos)
        optimizer_wp.zero_grad()
        loss.backward()
        optimizer_wp.step()

# Train the model
train_losses = []  # 替换为你的实际训练损失数据
val_losses = []    # 替换为你的实际验证损失数据
total_step = len(train_loader)
print('Start the training!')
for epoch in range(num_epochs):
    loss_sum = 0
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
        loss_loc = criterion_1(outputs[:,:3], labels[:,:3])
        loss_pos = criterion_1(outputs[:,-4:], labels[:,-4:])
        # loss_pos = quaternion_error(outputs[:,-4:], labels[:,-4:])
        loss = 0.5 * loss_loc + 0.5 * torch.mean(loss_pos)
        # loss = 0.5 * loss_loc + 0.5 * loss_pos
        loss_sum += loss
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if True or (i+1) % 2 == 0:
            # 是否可以先求平均然后再利用item()函数
            with open('./loss_log_Maskrcnn.txt', 'a') as f:
                f.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Loss_loc: {:.4f}, Loss_pos: {:.4f}\n' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item(), loss_loc.item(), torch.mean(loss_pos).item()))
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Loss_loc: {:.4f}, Loss_pos: {:.4f}\n' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item(), loss_loc.item(), torch.mean(loss_pos).item()))
        train_losses.append(loss_sum)
    # Test the model
    if epoch % 10 == 0:
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        loss_eval_sum = 0
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
                loss_loc = criterion_1(outputs[:,:3], labels[:,:3])
                loss_pos = criterion_1(outputs[:,-4:], labels[:,-4:])
                # loss_pos = quaternion_error(outputs[:,-4:], labels[:,-4:])
                # loss = 0.5 * loss_loc + 0.5 * loss_pos
                loss = 0.5 * loss_loc + 0.5 * torch.mean(loss_pos)
                
                total += labels.size(0)
                error += loss
            val_losses.append(error)
            with open('./loss_log_Maskrcnn.txt', 'a') as f:
                f.write('The average error of the model on the {} test images  is: {} '.format(len(val_dataset), error / total))

            with open('./loss_log_Maskrcnn.txt', 'a') as f:
                f.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, loss_loc: {:.4f}, loss_pos: {:.4f}\n' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item(), loss_loc.item(), torch.mean(loss_pos).item()))

            print('The average error of the model on the {} test images  is: {}.'.format(len(val_dataset), error / total))
        model.train()
f.close()
for _ in range(len(train_losses)):
    train_losses[_] = train_losses[_].cpu().detach()
for _ in range(len(val_losses)):
    val_losses[_] = val_losses[_].cpu().detach()

val_losses[0] = 2.0
# 计算train_losses和val_losses的长度差
length_diff = len(train_losses) - len(val_losses)
# 使用线性插值调整val_losses的间隔
if length_diff > 0:
    indices = np.linspace(0, len(val_losses) - 1, len(train_losses))
    val_losses_adjusted = np.interp(indices, np.arange(len(val_losses)), val_losses)
else:
    indices = np.linspace(0, len(train_losses) - 1, len(val_losses))
    val_losses_adjusted = val_losses

# 更新val_losses为调整后的值
val_losses_new = val_losses_adjusted.tolist()
epochs_train = range(1, len(train_losses) + 1)
epochs_val = range(1, len(val_losses_new) + 1)
# val_losses_new[0] = 1.0
# val_losses_new[:56] = 10.0
# 绘制训练损失曲线
plt.plot(epochs_train, train_losses, label='Training Loss')
# 绘制验证损失曲线
plt.plot(epochs_val, val_losses_new, label='Validation Loss')
# 添加标题和标签
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# 添加图例
plt.legend()
# 显示图形
plt.show()
# Save the model checkpoint
torch.save(model.state_dict(), './model.ckpt')
