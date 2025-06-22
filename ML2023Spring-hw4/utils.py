import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SoftLabelCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftLabelCrossEntropyLoss, self).__init__()

    def forward(self, pred, soft_target):
        log_prob = F.log_softmax(pred, dim=1)              # 对预测结果做 log_softmax
        loss = -torch.sum(soft_target * log_prob, dim=1)   # 逐样本交叉熵
        return loss.mean()                                 # 求平均作为最终损失

class MetricComputer:
    def __init__(self):
        self.ce = nn.CrossEntropyLoss()
        self.ce_ls = SoftLabelCrossEntropyLoss() 

        # 分别记录训练和验证的loss和acc的总和及计数
        self.loss_sum_train = 0
        self.acc_sum_train = 0
        self.loss_num_train = 0
        self.acc_num_train = 0

        self.loss_sum_val = 0
        self.acc_sum_val = 0
        self.loss_num_val = 0
        self.acc_num_val = 0

    def loss_cpt(self, yp, y, mode):
        loss = self.loss_cpt_(yp, y)
        if mode == 'train':
            self.loss_sum_train += loss.item()
            self.loss_num_train += 1
        elif mode == 'val':
            self.loss_sum_val += loss.item()
            self.loss_num_val += 1 
        return loss

    def loss_cpt_(self, yp, y):
        if isinstance(yp, list):
            total_loss = 0.0
            for i in len(yp):
                if len(y.shape) == 1:  # [b]
                    loss = self.ce(ypi, y)
                else:                  # [b, n]
                    loss = self.ce_ls(ypi, y)
                if i == 0:
                    total_loss += loss  # 第一项权重为1
                else:
                    total_loss += 0.3 * loss  # 后续项权重为0.3
            return total_loss
        else:
            if len(y.shape) == 1:  # [b]
                return self.ce(yp, y)
            else:                  # [b, n]
                return self.ce_ls(yp, y)

    def acc_cpt(self, yp, y, mode):
        accuracy = self.acc_cpt_(yp, y)
        if mode == 'train':
            self.acc_sum_train += accuracy
            self.acc_num_train += 1
        elif mode == 'val':
            self.acc_sum_val += accuracy
            self.acc_num_val += 1
        return accuracy

    def acc_cpt_(self, yp, y):
        pred = torch.argmax(yp, dim=1)  # 预测类别
        if len(y.shape) == 1:  # [b]
            labels = y
        else:                  # [b, n]
            labels = torch.argmax(y, dim=1)  # 概率分布标签转类别
        correct = (pred == labels).sum().item()
        total = labels.shape[0]
        accuracy = 100 * correct / total
        return accuracy

    def get_loss(self, mode):
        if mode == 'train':
            return self.loss_sum_train / self.loss_num_train if self.loss_num_train > 0 else 0
        elif mode == 'val':
            return self.loss_sum_val / self.loss_num_val if self.loss_num_val > 0 else 0

    def get_acc(self, mode):
        if mode == 'train':
            return self.acc_sum_train / self.acc_num_train if self.acc_num_train > 0 else 0
        elif mode == 'val':
            return self.acc_sum_val / self.acc_num_val if self.acc_num_val > 0 else 0

    def reset(self, mode='all'):
        if mode == 'all':
            self.loss_sum_train = 0
            self.acc_sum_train = 0
            self.loss_num_train = 0
            self.acc_num_train = 0

            self.loss_sum_val = 0
            self.acc_sum_val = 0
            self.loss_num_val = 0
            self.acc_num_val = 0
        elif mode == 'train':
            self.loss_sum_train = 0
            self.acc_sum_train = 0
            self.loss_num_train = 0
            self.acc_num_train = 0
        elif mode == 'val':
            self.loss_sum_val = 0
            self.acc_sum_val = 0
            self.loss_num_val = 0
            self.acc_num_val = 0