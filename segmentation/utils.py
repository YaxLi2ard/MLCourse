import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class MetricComputer:
    def __init__(self, num_classes=21, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='mean')
        # 分别记录训练和验证的loss和各项指标 {'loss_sum', 'loss_num', 'pa_sum', 'pa_num', 'iou_sum', 'iou_num'}
        self.log = {'train': {}, 'val': {}}
        self.reset('all')

    def cpt_update(self, yp, y, mode):
        # yp: [b, 21, h, w] y: [b, 1, h, w] 计算yp和y的交叉熵损失、ap、iou 若y和yp大小不一致则把 yp 最近邻插值到y的大小
        # probs = torch.nn.functional.softmax(yp, dim=1)
        # print("Softmax min:", probs.min().item(), "max:", probs.max().item())
        if yp.shape[-2:] != y.shape[-2:]:
            yp = F.interpolate(yp, size=y.shape[-2:], mode='nearest')
        loss = self.loss_cpt(yp, y, mode)
        mpa = self.mpa_cpt(yp, y, mode)
        miou = self.miou_cpt(yp, y, mode)
        return loss, mpa, miou

    def loss_cpt(self, yp, y, mode):
        loss = self.loss_cpt_(yp, y)
        self.log[mode]['loss_sum'] += loss.item() * y.shape[0]
        self.log[mode]['loss_num'] += y.shape[0]
        return loss

    def loss_cpt_(self, yp, y):
        # 计算yp和y的交叉熵损失
        y = y.squeeze(1).long()
        loss = self.ce(yp, y)
        return loss
        
        # y = y.squeeze(1).long()
        # loss1 = F.cross_entropy(yp, y, ignore_index=255, reduction='none')
        # valid_mask = (y != 255).float()
        # print((loss1).dtype)
        # print((loss1 * valid_mask).dtype)
        # loss1 = (loss1 * valid_mask).sum() / valid_mask.sum()
        # print(loss1.item())

        # loss2 = F.cross_entropy(yp, y, ignore_index=255, reduction='mean')
        # print(loss2.item())
        # return loss2
    
    def mpa_cpt(self, yp, y, mode):
        pa = self.pa_cpt_(yp, y)
        self.log[mode]['pa_sum'] += pa
        self.log[mode]['pa_num'] += y.shape[0]
        return pa / y.shape[0]

    def pa_cpt_(self, yp, y):
        # 计算yp和y的pixel acc batch内总和
        pred = yp.argmax(dim=1)  # [B, H, W]
        target = y.squeeze(1)  # [B, H, W]
        # 有效位置 mask：忽略 ignore_index
        valid = (target != self.ignore_index)  # [B, H, W]
        # 预测正确的 mask
        correct = (pred == target) & valid  # [B, H, W]
        # 每个样本的有效像素数 [B]
        valid_count = valid.view(y.shape[0], -1).sum(dim=1)  # [B]
        # 每个样本预测正确的像素数 [B]
        correct_count = correct.view(y.shape[0], -1).sum(dim=1)  # [B]
        # 防止除以 0：有效像素为 0 的样本跳过
        pa = torch.where(valid_count > 0, correct_count.float() / valid_count.float(),
                         torch.zeros_like(valid_count, dtype=torch.float))
        return pa.sum().item()  # 返回 batch 内所有样本 pixel acc 的总和

    def miou_cpt(self, yp, y, mode):
        miou = self.miou_cpt_(yp, y)
        self.log[mode]['miou_sum'] += miou
        self.log[mode]['miou_num'] += y.shape[0]
        return miou / y.shape[0]

    def miou_cpt_(self, yp, y):
        # 计算yp(argmax后)和y的iou batch内总和
        pred = yp.argmax(dim=1)  # [B, H, W]
        target = y.squeeze(1)  # [B, H, W]
        B = target.shape[0]
        iou_list = []
        for b in range(B):
            pred_b = pred[b]  # [H, W]
            target_b = target[b]  # [H, W]
            valid_mask = (target_b != self.ignore_index)
            iou_sum = 0.0
            valid_class_count = 0
            for cls in range(self.num_classes):
                pred_cls = (pred_b == cls)
                target_cls = (target_b == cls)
                # 排除 ignore 区域
                pred_cls = pred_cls & valid_mask
                target_cls = target_cls & valid_mask
                intersection = (pred_cls & target_cls).sum().item()
                union = (pred_cls | target_cls).sum().item()
                if target_cls.sum() == 0:
                    continue  # 当前样本中没有该类，跳过
                iou = intersection / union
                iou_sum += iou
                valid_class_count += 1
            if valid_class_count > 0:
                iou_list.append(iou_sum / valid_class_count)
        return sum(iou_list)

    def get_metric(self, mode):
        return {
            'loss': self.get_loss(mode),
            'mpa': self.get_mpa(mode),
            'miou': self.get_miou(mode)
        }

    def get_loss(self, mode):
        return self.log[mode]['loss_sum'] / self.log[mode]['loss_num']

    def get_mpa(self, mode):
        return self.log[mode]['pa_sum'] / self.log[mode]['pa_num']

    def get_miou(self, mode):
        return self.log[mode]['miou_sum'] / self.log[mode]['miou_num']

    def reset(self, mode='all'):
        if mode == 'all':
            self.reset('train')
            self.reset('val')
        else:
            self.log[mode]['loss_sum'] = 0.0
            self.log[mode]['loss_num'] = 0
            self.log[mode]['pa_sum'] = 0.0
            self.log[mode]['pa_num'] = 0
            self.log[mode]['miou_sum'] = 0.0
            self.log[mode]['miou_num'] = 0

class TransformTrain:
    def __init__(self):
        self.transform = A.Compose([
            A.Resize(height=256, width=256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=(0, 0.3), rotate_limit=15, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()  # 转为 PyTorch Tensor
        ])

    def __call__(self, image, mask):
        image = np.array(image)  # PIL -> ndarray
        mask = np.array(mask)    # PIL -> ndarray, 值为 class index (0~20 或 255)
        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented['image']           # FloatTensor, [3, H, W]
        mask_tensor = augmented['mask'].long()      # LongTensor, [H, W]
        return image_tensor, mask_tensor.unsqueeze(0)  # [3, H, W], [1, H, W]

class TransformTest:
    def __init__(self, label_resize=False):
        self.label_resize = label_resize
        self.transform = A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()  # 转为 PyTorch Tensor
        ])

    def __call__(self, image, mask):
        image = np.array(image)  # PIL -> ndarray
        mask = np.array(mask)    # PIL -> ndarray, 值为 class index (0~20 或 255)
        if self.label_resize:
            transformed = self.transform(image=image, mask=mask)
            image_tensor = transformed['image']
            mask_tensor = transformed['mask']
        else:  # 否则mask只totensor
            transformed = self.transform(image=image)
            image_tensor = transformed['image']
            mask_tensor = torch.from_numpy(mask)
        return image_tensor, mask_tensor.long().unsqueeze(0)

class TransformDual:
    def __init__(self, transform, label_trans=False):
        self.label_trans = label_trans
        self.transform = transform

    def __call__(self, image, mask):
        image = np.array(image)  # PIL -> ndarray
        mask = np.array(mask)    # PIL -> ndarray, 值为 class index (0~20 或 255)
        if self.label_trans:
            transformed = self.transform(image=image, mask=mask)
            image_tensor = transformed['image']
            mask_tensor = transformed['mask']
        else:  # 否则mask只totensor
            transformed = self.transform(image=image)
            image_tensor = transformed['image']
            mask_tensor = torch.from_numpy(mask)
        return image_tensor, mask_tensor.long().unsqueeze(0)

def voc_colormap(N=21):
    """
    返回 VOC 的调色板，shape [N, 3]，每个类别一个 RGB 色
    """
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        cid = i
        for j in range(8):
            r |= bitget(cid, 0) << (7 - j)
            g |= bitget(cid, 1) << (7 - j)
            b |= bitget(cid, 2) << (7 - j)
            cid >>= 3
        cmap[i] = [r, g, b]
    return cmap

def decode_segmap(label_mask, colormap):
    """
    label_mask: [H, W] numpy array of int (0~20 or 255)
    colormap: [N, 3] 的 VOC 调色板
    return: [H, W, 3] RGB 彩图
    """
    h, w = label_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx in np.unique(label_mask):
        if class_idx == 255:
            color_mask[label_mask == 255] = colormap[21]  # index=21 表示 ignore
            continue  # ignore 类别不显示
        else:
            color_mask[label_mask == class_idx] = colormap[class_idx]
    return color_mask

def visualize_image_mask_overlay(x, y, alpha=0.5):
    """
    x: torch.Tensor, [3,H,W], 图像tensor
    y: torch.Tensor, [1,H,W], 掩码类别索引
    alpha: 掩码透明度，0~1，越大掩码越明显
    """
    # 反归一化图像
    mean = torch.tensor([0.456, 0.443, 0.409]).view(3,1,1)
    std = torch.tensor([0.231, 0.227, 0.233]).view(3,1,1)
    x = x.cpu() * std + mean
    x = x.clamp(0,1)
    x_np = x.permute(1,2,0).numpy()

    # 掩码转numpy
    y_np = y.squeeze(0).cpu().numpy()

    # 生成掩码颜色图
    cmap = voc_colormap(22)
    mask_color = decode_segmap(y_np, cmap)  # [H, W, 3], uint8
    mask_color = mask_color.astype(np.float32) / 255.0  # 转成 [0,1] 范围才能混合

    # 叠加：img * (1 - alpha) + mask_color[..., :3] * alpha
    overlay = x_np * (1 - alpha) + mask_color[..., :3] * alpha
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(6,6))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title("Image with Mask Overlay")
    plt.show()

def get_image_mask_overlay(x, y, alpha=0.5):
    """
    x: torch.Tensor, [3,H,W], 图像tensor
    y: torch.Tensor, [1,H,W], 掩码类别索引
    alpha: 掩码透明度，0~1，越大掩码越明显
    """
    # 反归一化图像
    mean = torch.tensor([0.456, 0.443, 0.409]).view(3,1,1)
    std = torch.tensor([0.231, 0.227, 0.233]).view(3,1,1)
    x = x.cpu() * std + mean
    x = x.clamp(0,1)
    x_np = x.permute(1,2,0).numpy()

    # 掩码转numpy
    y_np = y.squeeze(0).cpu().numpy()

    # 生成掩码颜色图
    cmap = voc_colormap(22)
    mask_color = decode_segmap(y_np, cmap)  # [H, W, 3], uint8
    mask_color = mask_color.astype(np.float32) / 255.0  # 转成 [0,1] 范围才能混合

    # 叠加：img * (1 - alpha) + mask_color[..., :3] * alpha
    overlay = x_np * (1 - alpha) + mask_color[..., :3] * alpha
    overlay = np.clip(overlay, 0, 1)

    return overlay