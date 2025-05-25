import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import random
from model.AlexNet import *
from model.ResNet import *
from model.ResNet_a import *
from model.VGGNet import *
from dataset import *
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
lr = 0.0001
batch_size = 96
epochs = 999

# dataset and dataloader
img_sz = 256
normalize_mean = [0.558034, 0.45292863, 0.3462093]
normalize_std = [0.22920622, 0.23979892, 0.23905471]
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(img_sz, img_sz), scale=(0.9, 1.0)),
    # transforms.Resize(size=(256, 256)),
    transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=60),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mean, std=normalize_std)
])
transform_test = transforms.Compose([
    transforms.Resize(size=(img_sz, img_sz)),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mean, std=normalize_std)
])

data_train = '/root/autodl-tmp/MLCourseDataset/ml2023spring-hw3/train'
data_val = '/root/autodl-tmp/MLCourseDataset/ml2023spring-hw3/valid'
data_test = '/root/autodl-tmp/MLCourseDataset/ml2023spring-hw3/test'
tta = 0
dataset_train = FoodClsDataset(data_dir=data_train, mode='train', tta=0, transform=transform_train)
dataset_val = FoodClsDataset(data_dir=data_val, mode='val', tta=0, transform=transform_test)
dataset_test = FoodClsDataset(data_dir=data_test, mode='test', tta=0, transform=transform_test)

num_workers = 16
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)


# 网络、优化器、计算器
model = resnet50(num_classes=11)
model.to(device)
optimizer=torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer=optimizer,
    step_size=5,   # 每隔一定步学习率衰减一次
    gamma=0.9         # 衰减系数
)
metric_cpt = MetricComputer()
scaler = GradScaler()
