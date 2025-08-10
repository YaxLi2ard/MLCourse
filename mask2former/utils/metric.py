import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MetricLogger:
    def __init__(self):
        self.log = {}
        
    def update(self, k, value, num=1):
        self.update_(k, value)
        self.update_(k+'_num', num)
        
    def update_(self, k, value):
        if k in self.log:
            self.log[k] += value
        else:
            self.log[k] = value

    def get_metric(self, k):
        return self.log[k] / self.log[k+'_num']
    
    def reset(self, k):
        if k == 'all' or k == 'train' or k == 'val':
            for key in self.log.keys():
                if k == 'all':
                    self.reset(key)
                else:
                    if k in key:
                        self.reset(key)
        else:
            self.log[k] = 0
            self.log[k+'_num'] = 0


class MetricCpt:
    def __init__(self, num_classes, ignore_id=255, device='cpu'):
        """
        语义分割评估指标计算类，维护混淆矩阵，计算 mIoU 和 mDice
        
        Args:
            num_classes (int): 类别数
            ignore_id (int): 需要忽略的标签id
            device (str or torch.device): 设备
        """
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.device = device
        self.reset()
        
    def reset(self):
        """重置混淆矩阵"""
        self.conf_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64, device=self.device
        )
    
    def _fast_hist(self, label_true, label_pred):
        """
        计算单批混淆矩阵，忽略ignore_id
        Args:
            label_true: [N, H, W] 或 [H, W]，ground truth
            label_pred: 同 shape，预测结果
        Returns:
            conf_matrix: [num_classes, num_classes]的混淆矩阵
        """
        # 展平
        label_true = label_true.view(-1)
        label_pred = label_pred.view(-1)
        
        # 过滤ignore_id
        mask = label_true != self.ignore_id
        label_true = label_true[mask]
        label_pred = label_pred[mask]
        
        # 计算混淆矩阵索引，就是每个像素的分类情况（真实标签和预测的情况）应该加载混淆矩阵的哪个位置
        idx = label_true * self.num_classes + label_pred
        # 统计每个类别对的出现次数
        hist = torch.bincount(idx, minlength=self.num_classes ** 2)
        hist = hist.reshape(self.num_classes, self.num_classes)
        return hist
    
    def update(self, preds, labels):
        """
        更新混淆矩阵
        
        Args:
            preds: Tensor, [N,H,W]或[H,W]，预测类别标签
            labels: Tensor, 同shape，真实类别标签
        """
        if preds.dim() == 2:
            preds = preds.unsqueeze(0)
            labels = labels.unsqueeze(0)
        
        for pred, label in zip(preds, labels):
            hist = self._fast_hist(label, pred)
            self.conf_matrix += hist
    
    def compute(self):
        """
        根据混淆矩阵计算 mIoU 和 mDice
        
        Returns:
            miou: float，mean IoU
            mdice: float，mean Dice
        """
        conf = self.conf_matrix.float()
        
        TP = torch.diag(conf)  # [num_classes]
        FP = conf.sum(dim=0) - TP
        FN = conf.sum(dim=1) - TP
        # GT中该类的像素数
        gt_count = conf.sum(dim=1)
        
        # 计算 IoU：TP / (TP + FP + FN)
        denom_iou = TP + FP + FN
        iou = torch.where(denom_iou > 0, TP / denom_iou, torch.tensor(float('nan'), device=self.device))
        
        # 计算 Dice：2*TP / (2*TP + FP + FN)
        denom_dice = 2 * TP + FP + FN
        dice = torch.where(denom_dice > 0, 2 * TP / denom_dice, torch.tensor(float('nan'), device=self.device))
        
        # 只对 GT > 0 的类别计算平均，忽略nan
        valid = gt_count > 0
        miou = torch.nanmean(iou[valid]).item()
        mdice = torch.nanmean(dice[valid]).item()
        
        return miou, mdice
        
    
