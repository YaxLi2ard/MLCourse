import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import random
from model.Transformer import Transformer
from dataset import *
from tools.tokenize_ import Tokenizer
from utils import *
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

# 超参数
lr = 0.001
batch_size = 91
epochs = 999

# dataset and dataloader
train_src = './DATA/rawdata/split/train_ids.en'
train_tgt = './DATA/rawdata/split/train_ids.zh'
dataset_train = TranslationDataset(train_src, train_tgt, max_len=100)
val_src = './DATA/rawdata/split/val_ids.en'
val_tgt = './DATA/rawdata/split/val_ids.zh'
dataset_val = TranslationDataset(val_src, val_tgt, max_len=100)

num_workers = 16
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False, num_workers=num_workers, pin_memory=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False, num_workers=num_workers, pin_memory=True)

# tokenizer
src_tokenizer = Tokenizer("./tools/ted2020.model")
tgt_tokenizer = Tokenizer("./tools/ted2020.model")  # 英文和中文用同一模型

# 网络、优化器、计算器
model = Transformer(src_vocab_size=src_tokenizer.vocab_size(), tgt_vocab_size=tgt_tokenizer.vocab_size())
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)

def lr_lambda(step):
    # 第warmup_steps轮开始时step为warmup_steps-1
    # warmup设置为1即取消
    warmup_steps = 3
    if step < warmup_steps:
        return (step + 1) / warmup_steps  # 线性 warmup
    return max(0.9 ** ((step + 1 - warmup_steps) // 1), 0.1)
    
# scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer=optimizer,
#     step_size=5,   # 每隔一定步学习率衰减一次
#     gamma=0.9         # 衰减系数
# )
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

metric_cpt = MetricComputer()
scaler = GradScaler()
