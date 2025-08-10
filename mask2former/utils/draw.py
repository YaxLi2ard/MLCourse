import torch
from collections import namedtuple
import numpy as np

def get_cityscapes_colormap(labels, num_classes=19):
    """
    根据 Cityscapes label 列表，生成 trainId 到颜色的映射表
    
    返回:
        colormap: np.ndarray, shape=(num_classes, 3), dtype=np.uint8
    """
    # 初始化为黑色
    colormap = np.zeros((num_classes, 3), dtype=np.uint8)
    
    # 用于标记 trainId 是否已被赋色（防止重复赋值）
    assigned = np.zeros(num_classes, dtype=bool)
    
    for label in labels:
        trainId = label.trainId
        if 0 <= trainId < num_classes and not assigned[trainId]:
            colormap[trainId] = label.color
            assigned[trainId] = True
    
    return colormap


def decode_segmap(label_mask, colormap):
    """
    根据类别索引和颜色映射，生成彩色掩码图
    """
    h, w = label_mask.shape
    mask_color = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(colormap):
        mask_color[label_mask == class_id] = color
    return mask_color

def get_image_mask_overlay(x, y, img_mean, img_std, colormap, alpha=0.5):
    """
    x: torch.Tensor, [3,H,W], 图像tensor
    y: torch.Tensor, [H,W], 掩码类别索引
    alpha: 掩码透明度，0~1，越大掩码越明显
    """
    # 反归一化图像
    mean = torch.tensor(img_mean).view(3,1,1)
    std = torch.tensor(img_std).view(3,1,1)
    x = x.cpu() * std + mean
    x = x.clamp(0,1)
    x_np = x.permute(1,2,0).numpy()

    # 掩码转numpy
    y_np = y.cpu().numpy()

    # 用颜色映射解码掩码
    mask_color = decode_segmap(y_np, colormap)  # uint8 [H,W,3]
    mask_color = mask_color.astype(np.float32) / 255.0  # 转成 [0,1]
    
    # 叠加图像和掩码
    overlay = x_np * (1 - alpha) + mask_color * alpha
    overlay = np.clip(overlay, 0, 1)
    
    return overlay