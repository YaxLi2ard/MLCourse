import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
import random
from model.seg_model import SegModel
from dataset.VOCSegDataset import VOCSegDataset, VOC_COLORMAP
from dataset.ImgAugment import *
from utils.criterion import SetCriterion, FocalLoss
from utils.matcher import HungarianMatcher
from utils.metric import MetricLogger, MetricCpt
from utils.draw import get_image_mask_overlay
from tqdm import tqdm
import time
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

''' 参数 '''
lr = 0.0001
batch_size = 16
epochs = 999
num_workers = 16
pin_memory = True

''' dataset and dataloader '''
# ------------------------------------------------ voc ------------------------------------------------
img_sz = (320, 320)
normalize_mean = (0.456, 0.443, 0.409)
normalize_std = (0.231, 0.227, 0.233)
transform_train = A.Compose([
    RandomScaleAndCrop(scale_limit=(0.5, 2.0)),
    # ResizeToFit(target_h=img_sz[0], target_w=img_sz[1], p=1.0),
    A.Resize(height=img_sz[0], width=img_sz[1]),
    A.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.2, 0.2), p=0.6),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=normalize_mean, std=normalize_std),
    ToTensorV2()
])
transform_test = A.Compose([
    # ResizeToFit(target_h=512, target_w=512, p=1.0),
    A.Resize(height=320, width=320),
    A.Normalize(mean=normalize_mean, std=normalize_std),
    ToTensorV2()  # 转为 PyTorch Tensor
])

dataset_dir = '/root/autodl-tmp/MLCourseDataset/pascalvoc/VOCdevkit/VOC2012'
dataset_train = VOCSegDataset(root=dataset_dir, image_set='train', transform=transform_train)
dataset_val = VOCSegDataset(root=dataset_dir, image_set='val', transform=transform_test)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
# 掩码颜色映射
voc_colormap = VOC_COLORMAP
colormap = voc_colormap
num_classes = 21
# ------------------------------------------------ voc ------------------------------------------------

iter_val = iter(dataloader_val)
iter_train = iter(dataloader_train)
rng = np.random.default_rng()  # 独立生成器，不受 np.random.seed 影响
val_list = list(dataloader_val)
rng.shuffle(val_list)
iter_val = iter(val_list)
print('数据集加载成功...')


''' 网络、优化器 '''
model = SegModel(backbone='swin', head='maskformer', num_cls=21, pretrained=True)
model.to(device)
backbone_type = model.backbone_type
head_type = model.head_type

model.load_state_dict(torch.load('./cpt/ttt.pt'), strict=False)

print('模型加载成功...')


def set_lr(model, lr):
    # 所有参数名和参数
    backbone_params = []
    other_params = []
    for k, v in model.named_parameters():
        if 'backbone' in k:
            backbone_params.append(v)
        else:
            other_params.append(v)
    optim_params =  [
        {"params": other_params, "lr": lr},
        {"params": backbone_params, "lr": 0.1 * lr},
    ]
    return optim_params
    
params = set_lr(model, lr)
optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
# optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0001)
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
print('优化器加载成功...')

''' 损失计算器 '''
# 各个损失的权重
class_weight = 2
dice_weight = 5
mask_weight = 5
num_classes = 21  # 类别数量
num_points= 100
deep_supervision = True  # 是否深度监督，就是输出前面的一些层的输出是否计算损失，通常为True
no_object_weight = 0.1  # 损失中 背景类（无目标）类的权重，好像只有计算query分类时用到，mask相关的dice和bce没用到，默认0.1
common_stride = 4

# 二分图匹配
matcher = HungarianMatcher(
    cost_class=class_weight,
    cost_mask=mask_weight,
    cost_dice=dice_weight,
    num_points=num_points,
)
# 损失计算器
weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
if deep_supervision:
    dec_layers = 10
    aux_weight_dict = {}
    for i in range(dec_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
losses = ["labels", "masks"]
criterion = SetCriterion(
    num_classes,
    matcher=matcher,
    weight_dict=weight_dict,
    eos_coef=no_object_weight,
    losses=losses,
    num_points=num_points,
    oversample_ratio=3,
    importance_sample_ratio=0.75,
    ignore_id=255,
    device=device
)
criterion_ce = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
# criterion_ce = FocalLoss(gamma=2.0, alpha=0.25, ignore_index=255, reduction='mean')

# 指标记录器、指标计算器
metric_cpt = MetricCpt(num_classes=num_classes, ignore_id=255, device=device)
metric_logger = MetricLogger()
scaler = GradScaler()
print('损失计算器加载成功...')



