import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import segmentation_models_pytorch as segmodel
from torch.cuda.amp import autocast, GradScaler
import random
from model.FCN import FCN
# from model.UNet import UNet
from model.UNet_ResNet import UNet_ResNet
# from model.UNet_ import UNet_
from model.DeepLabV3 import DeepLabV3
# from model.DeepLabV3_ import DeepLabV3_
from VOCSegDataset import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
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
lr = 0.0001
batch_size = 32
epochs = 999
img_sz = (320, 320)
normalize_mean = (0.456, 0.443, 0.409)
normalize_std = (0.231, 0.227, 0.233)
num_workers = 16
pin_memory = True

# dataset and dataloader
transform_train = A.Compose([
     A.RandomResizedCrop(
        size=(img_sz[0], img_sz[1]),  # 输出尺寸
        scale=(0.7, 1.0),       # 裁剪区域占原图面积比例（最小70%，最大100%）
        ratio=(0.75, 1.33),     # 裁剪区域宽高比范围
        p=1.0
    ),
    # A.Resize(height=img_sz[0], width=img_sz[1]),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
    # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=(0, 0.3), rotate_limit=30, p=0.7),
    A.Normalize(mean=normalize_mean, std=normalize_std),
    ToTensorV2()  # 转为 PyTorch Tensor
])
transform_test = A.Compose([
    A.Resize(height=img_sz[0], width=img_sz[1]),
    A.Normalize(mean=normalize_mean, std=normalize_std),
    ToTensorV2()  # 转为 PyTorch Tensor
])
transform_train = TransformDual(transform_train, label_trans=True)
transform_test = TransformDual(transform_test, label_trans=False)

dataset_dir = '/root/autodl-tmp/MLCourseDataset/pascalvoc/VOCdevkit/VOC2012'
dataset_train = VOCSegDataset(root=dataset_dir, image_set='train', transform=transform_train)
dataset_val = VOCSegDataset(root=dataset_dir, image_set='val', transform=transform_test)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
iter_val = iter(dataloader_val)
iter_train = iter(dataloader_train)
rng = np.random.default_rng()  # 独立生成器，不受 np.random.seed 影响
val_list = list(dataloader_val)
rng.shuffle(val_list)
iter_val = iter(val_list)

# 网络、优化器、计算器
# model = FCN(num_classes=21, backbone='resnet50', pretrained=True)
# model = UNet(num_classes=21, base_c=64)
# model = UNet_ResNet(num_classes=21, backbone='resnet50', pretrained=True)
# model = UNet_(num_classes=21, backbone='resnet50', pretrained=True)
model = DeepLabV3(num_classes=21, backbone='resnet50', pretrained=True)
# model = DeepLabV3_(num_classes=21, backbone='resnet50', pretrained=True)
# print(segmodel.encoders.encoders.keys())
# model = segmodel.Unet(
#     encoder_name='resnet50',        # 编码器主干网络
#     encoder_weights='imagenet',     # 使用 ImageNet 预训练权重
#     in_channels=3,                  # 输入通道数
#     classes=21                      # 输出分类数
# )
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
def lr_lambda(step):
    # 第warmup_steps轮开始时step为warmup_steps-1
    # warmup设置为1即取消
    warmup_steps = 1
    if step < warmup_steps:
        return (step + 1) / warmup_steps  # 线性 warmup
    # return max(0.9 ** ((step + 1 - warmup_steps) // 5), 0.1)
    return (1 - ((step + 1 - warmup_steps) / (111 - warmup_steps))) ** 0.9

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

metric_cpt = MetricComputer()
scaler = GradScaler()