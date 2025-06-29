import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class SoftLabelCrossEntropyLoss(nn.Module):
#     def __init__(self):
#         super(SoftLabelCrossEntropyLoss, self).__init__()

#     def forward(self, pred, soft_target):
#         log_prob = F.log_softmax(pred, dim=1)              # 对预测结果做 log_softmax
#         loss = -torch.sum(soft_target * log_prob, dim=1)   # 逐样本交叉熵
#         return loss.mean()                                 # 求平均作为最终损失

class LabelSmoothedCrossEntropyCriterion(nn.Module):
    def __init__(self, smoothing, ignore_index=None, reduce=True):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, logits, target):
        lprobs = F.log_softmax(logits.float(), dim=-1)
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        # nll: Negative log likelihood 负对数似然，当目标是 one-hot 时的交叉熵。下一行代码等同于F.nll_loss
        nll_loss = -lprobs.gather(dim=-1, index=target)
        #  保留一些其他标签的概率，这样在计算交叉熵的时候相当于对所有标签的对数概率求和
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
            n_valid = (~pad_mask).sum().clamp_min(1).float()
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
            n_valid = logits.shape[0]
        if self.reduce:
            nll_loss = nll_loss.sum() / n_valid
            smooth_loss = smooth_loss.sum() / n_valid
        # 在计算交叉熵的时候，增加其他标签的损失
        eps_i = self.smoothing / lprobs.size(-1)
        loss = (1.0 - self.smoothing) * nll_loss + eps_i * smooth_loss
        return loss

class MetricComputer:
    def __init__(self):
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.ce_ls = LabelSmoothedCrossEntropyCriterion(smoothing=0.1, ignore_index=0)

        # 分别记录训练和验证的loss
        self.loss_sum_train = 0
        self.loss_num_train = 0

        self.loss_sum_val = 0
        self.loss_num_val = 0

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
        # 需要将y左移一位作为标签
        yp = yp[:, :-1, :]  # 去掉最后一位，不参与预测
        y = y[:, 1:]        # 去掉 <bos>，目标是预测下一个 token
        yp = yp.reshape(-1, yp.size(-1))  # [b, tgt_len, vocab_size] -> [b*tgt_len, vocab_size]
        y = y.reshape(-1)  # [b, tgt_len] -> [b*tgt_len]
        return self.ce_ls(yp, y)

    def get_loss(self, mode):
        if mode == 'train':
            return self.loss_sum_train / self.loss_num_train if self.loss_num_train > 0 else 0
        elif mode == 'val':
            return self.loss_sum_val / self.loss_num_val if self.loss_num_val > 0 else 0

    def reset(self, mode='all'):
        if mode == 'all':
            self.loss_sum_train = 0
            self.loss_num_train = 0

            self.loss_sum_val = 0
            self.loss_num_val = 0
        elif mode == 'train':
            self.loss_sum_train = 0
            self.loss_num_train = 0
        elif mode == 'val':
            self.loss_sum_val = 0
            self.loss_num_val = 0