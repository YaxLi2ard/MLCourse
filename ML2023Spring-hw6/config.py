import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import random
from dataset import ImgDataset
from model import load_model
import numpy as np
import os
import sys
import torchvision.utils as vutils
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
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
lr_gen = 0.001
lr_dis = 0.001
epochs = 999
data_dir = '/root/autodl-tmp/MLCourseDataset/ml2023spring-hw6/faces/faces'
img_sz = 64
batch_size = 16
pin_memory = True
num_workers = 16
gen_zdim = 512
accum_steps = 4 # 梯度累计步数
use_ema = True
ema_decay = 0.995
continue_train = False

# dataset and dataloader
dataset = ImgDataset(data_dir, img_sz)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)


# 网络、优化器
gen = load_model.load_generator(img_sz, pretrained=False).to(device)
dis = load_model.load_discriminator(img_sz, pretrained=False).to(device)
# dis = load_model.load_discriminator_resnet(img_sz, pretrained=True).to(device)
opt_gen = torch.optim.Adam(gen.parameters(), lr=lr_gen, weight_decay=0.0000, betas=(0., 0.99))
opt_dis = torch.optim.Adam(dis.parameters(), lr=lr_dis, weight_decay=0.0000, betas=(0.9, 0.99))

if continue_train:
    gen.load_state_dict(torch.load('./cpt/gen_ema1.pt', map_location=device), strict=False)
    dis.load_state_dict(torch.load('./cpt/dis1.pt', map_location=device), strict=False)


def lr_lambda(step):
    # 第warmup_steps轮开始时step为warmup_steps-1
    # warmup设置为1即取消
    warmup_steps = 500
    if step < warmup_steps:
        return (step + 1) / warmup_steps  # 线性 warmup
    return max(1 ** ((step + 1 - warmup_steps) // 300), 0.1)

scheduler_gen = torch.optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda)
scheduler_dis = torch.optim.lr_scheduler.LambdaLR(opt_dis, lr_lambda)

# scaler = GradScaler()


