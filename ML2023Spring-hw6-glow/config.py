import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import random
from dataset import ImgDataset
from model.glow import Glow
import numpy as np
import os
import math
import sys
import torchvision.utils as vutils
import time
from tqdm import tqdm

writer = SummaryWriter(log_dir="/root/tf-logs")

def seed_everything(seed=9):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed_everything(seed=999)


# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 参数
# train
lr = 0.0003
start_iter = 0
iters = 100000
data_dir = '/root/autodl-tmp/MLCourseDataset/ml2023spring-hw6/faces/faces'
# data_dir = 'Y:/Dataset/动漫人脸生成/faces/faces'
img_sz = 64
batch_size = 61

# model
n_bits = 5  # 图像离散位数（例如 8 位）
n_bins = 2.0 ** n_bits  # 图像离散等级（例如 2^8 = 256）
n_flow = 32
n_block = 4
affine = True
no_lu = True

# test
temperature = 0.7  # 生成时的温度系数
n_sample = 9  # 每次生成的样本数

# dataloader
pin_memory = True
num_workers = 16
use_ema = True
ema_decay = 0.995
continue_train = False

# dataset and dataloader
dataset = ImgDataset(data_dir, img_sz)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
class IterLoader:
    '''
    迭代数据加载器，使得迭代器在数据用尽后重新初始化
    '''
    def __init__(self, dataloader):
        self.dataloader = dataloader  # 存储数据加载器dataloader
        self.iter = iter(self.dataloader)  # 初始化迭代器

    def next_one(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)  # 重新初始化迭代器
            return next(self.iter)  # 返回新的样本
loader = IterLoader(dataloader)

# 网络、优化器
model = Glow(
    in_channel=3,
    n_flow=n_flow,
    n_block=n_block,
    affine=affine,
    conv_lu=not no_lu
).to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0000, betas=(0.9, 0.999))



def lr_lambda(step):
    # 第warmup_steps轮开始时step为warmup_steps-1
    # warmup设置为1即取消
    warmup_steps = 300
    if step < warmup_steps:
        return (step + 1) / warmup_steps  # 线性 warmup
    return max(1 ** ((step + 1 - warmup_steps) // 500), 0.3)

scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

scaler = GradScaler()

if continue_train:
    print('加载断点处状态...')
    checkpoint = torch.load('cpt/cpt.pt')
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])  # 恢复状态
    start_iter = checkpoint['iter']
    
