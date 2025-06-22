import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import random
from model.Conformer import Conformer
from model.Conformer2 import Conformer2
from model.Transformer import Transformer
from model.SwinTransformer import SwinTransformer
from DatasetManager import *
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
batch_size = 61
epochs = 999

# dataset and dataloader
data_dir = '/root/autodl-tmp/MLCourseDataset/ml2023spring-hw4/Dataset'
dataset_manager = DatasetManager(data_dir, valid_ratio=0.1, segment_len=128)
dataset_train, dataset_val, dataset_val2, dataset_test, dataset_test2 = dataset_manager.get_datasets()
id2speaker = dataset_manager.get_id2speaker()

num_workers = 16
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)


# 网络、优化器、计算器
# model = Conformer()
# model = Conformer2()
model = Transformer()
# model = SwinTransformer()
model.to(device)
optimizer=torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)
def lr_lambda(step):
    # 第warmup_steps轮开始时step为warmup_steps-1
    # warmup设置为1即取消
    warmup_steps = 5
    if step < warmup_steps:
        return (step + 1) / warmup_steps  # 线性 warmup
    return max(0.9 ** ((step + 1 - warmup_steps) // 5), 0.1)

# scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer=optimizer,
#     step_size=5,   # 每隔一定步学习率衰减一次
#     gamma=0.96         # 衰减系数
# )
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

metric_cpt = MetricComputer()
scaler = GradScaler()
