# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
import sys
import os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from .point_features import point_sample, get_uncertain_point_coords_with_randomness
from .misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list, get_world_size


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    计算 DICE Loss，用于衡量两个二值掩码之间的相似度，类似于用于mask的广义IOU。
    参数说明：
        inputs: torch.Tensor，预测值张量，任意形状，通常为模型输出的掩码预测（未sigmoid激活）。
        targets: torch.Tensor，与inputs相同形状的目标张量，二值标签（0表示背景，1表示前景）。
        num_masks: float，参与loss计算的mask数量。
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    计算带 sigmoid 的二值交叉熵损失（Binary Cross Entropy with Logits），
    用于对每个像素或元素进行二分类的损失计算。
    参数：
        inputs: torch.Tensor，预测值张量，任意形状，通常为模型输出（未经过 sigmoid 激活）。
        targets: torch.Tensor，目标标签，与 inputs 形状相同，取值为 0（背景）或 1（前景）。
        num_masks: float，参与损失计算的 mask 数量，用于归一化最终 loss。

    返回：
        单个标量 tensor，表示归一化后的总损失。
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks

def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """
    该类用于计算 Mask2Former 模型的训练损失。
    整体流程包括两个主要步骤：
    1. 使用匈牙利算法（Hungarian algorithm）将模型预测结果和真实目标进行匹配；
    2. 针对匹配的每对预测-标签，对类别、边界框或 mask 等执行监督，计算对应的损失函数。
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, ignore_id, device):
        """
        初始化 SetCriterion 损失模块。
        参数说明：
        - num_classes: int，类别数，不包含特殊的 no-object 类别（一般是背景类）。
        - matcher: nn.Module，负责计算预测和真实标签之间的最佳匹配（通常使用匈牙利算法）。
        - weight_dict: dict，损失项权重的字典，比如 {"loss_ce": 1, "loss_mask": 5, "loss_dice": 5}。
        - eos_coef: float，用于控制 no-object 类别（背景类）在分类损失中的权重，常设为一个较小值（如0.1）。
        - losses: list，指定使用哪些损失函数（字符串列表），如 ["labels", "masks"]。
        - num_points: int，计算 mask loss 时采样的点数。
        - oversample_ratio: float，点采样时的过采样比率（>1 会先采样多一些再筛选）
        - importance_sample_ratio: float，重要性采样的比例，控制多少比例的点是从高置信区域中采样。
        - device: 设备信息（'cuda' 或 'cpu'），用于 tensor 的移动与注册。
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.ignore_id = ignore_id
        self.device = device
        # 初始化分类损失的类别权重向量（类别数 + 1 是因为加入了 no-object 类）
        empty_weight = torch.ones(self.num_classes + 1).to(device)  # [类别1, 类别2, ..., 背景类]
        empty_weight[-1] = self.eos_coef  # 为最后一类（背景）设置较小的权重
        self.register_buffer("empty_weight", empty_weight)  # 将其注册为 buffer，参与模型保存但不更新梯度

        # point-wise mask 损失相关参数，用于加速 mask loss 计算（点采样代替 full mask）
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """
        计算类别的交叉熵损失（cross-entropy loss）。
        参数：
        - outputs: 模型输出的字典，必须包含 "pred_logits"，形状为 [batch_size, num_queries, num_classes + 1]；
        - targets: 每张图像的标签列表，每个元素是一个 dict，至少包含键 "labels"，shape 为 [num_gt]；
        - indices: list，长度为 batch_size，每个元素是 (index_i, index_j) 的元组，表示该图像中预测与标签的匹配索引；
        - num_masks: 所有图像中目标框（或实例 mask）的总数，没用上。
        返回：
        - losses: dict，包含 "loss_ce" 这一项，表示分类损失。
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()  # 取出预测的 logits

        idx = self._get_src_permutation_idx(indices)
        # target_classes_o 类似于 tensor([1, 2, 3, 1, 2])，其中1，2是图像1的某2个query对应的类别，3，1，2是图像2的某3个query对应的类别
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(self.device)
        # 初始化所有预测的标签为0，shape = [bs, num_queries]，用于构建与该任务适配的 ground truth 标签
        # 博主原来self.num_classes的位置是0，应该是错误的
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        # 将匹配成功的预测位置的标签替换为真实标签，其余未匹配位置默认是0，代表 no-object（背景类）
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2),  # -> [bs, num_classes+1, num_queries]
                                  target_classes,
                                  self.empty_weight)  # 用于调整各类别的损失权重
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]  # [b, num_q, h, w]
        src_masks = src_masks[src_idx]  # 取出被匹配上的预测掩码，shape: [num_matched, h, w]
        masks = [t["masks"] for t in targets]  # # 列表， [num_mask, h, w]
        # TODO use valid to mask invalid areas due to padding in loss
        # 对 GT 掩码做了 nested_tensor_from_tensor_list()，是为了对 batch 内不同大小的掩码进行打包处理
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]  # # 取出被匹配上的 GT 掩码，shape: [num_matched, h, w]
        ''' 这样 src_masks 和 target_masks 形状为 [num_matched, h, w]，就是一一对应的了'''
        # ===================================================================================
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W

        # src_masks = src_masks[:, None]
        # target_masks = target_masks[:, None]

        # with torch.no_grad():
        #     # sample point_coords
        #     point_coords = get_uncertain_point_coords_with_randomness(
        #         src_masks,
        #         lambda logits: calculate_uncertainty(logits),
        #         self.num_points,
        #         self.oversample_ratio,
        #         self.importance_sample_ratio,
        #     )
        #     # get gt labels
        #     point_labels = point_sample(
        #         target_masks,
        #         point_coords,
        #         align_corners=False,
        #     ).squeeze(1)

        # point_logits = point_sample(
        #     src_masks,
        #     point_coords,
        #     align_corners=False,
        # ).squeeze(1)
        # ===================================================================================
        # 直接展平计算损失
        point_logits = src_masks.flatten(1)
        point_labels = target_masks.flatten(1)       

        losses = {
            "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks), # sigmoid_focal_loss(point_logits, point_labels, num_masks), # 
            "loss_dice": dice_loss(point_logits, point_labels, num_masks)
        }

        del src_masks
        del target_masks
        return losses

    '''
        indices = [
            (tensor([1, 4]), tensor([0, 2])),   # 第 0 张图像：第 1 和 4 个 query 匹配到了第 0 和 2 个 gt
            (tensor([0]), tensor([1]))          # 第 1 张图像：第 0 个 query 匹配到了第 1 个 gt
        ]
        return:
            batch_idx = tensor([0, 0, 1])  # 对应图像编号
            src_idx   = tensor([1, 4, 0])  # 对应被匹配的 query 编号
    '''
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    '''
    return:
        batch_idx = tensor([0, 0, 1])  # 对应图像编号
        src_idx   = tensor([0, 2, 1])  # 对应被匹配的 gt 编号
    '''
    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    # 将掩码转换为 one-hot 格式
    def _get_binary_mask(self, target):
        # target 的 shape 是 [H, W]，每个像素值是 0~num_classes 的类别编号
        y, x = target.size()  # 获取目标掩码的高和宽
        target_onehot = torch.zeros(self.num_classes + 1, y, x).to(target.device)  # 初始化 one-hot tensor，shape 为 [num_classes+1, H, W]
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)  # 特定位置、通道为1
        return target_onehot

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        # # 调用对应的损失函数并返回
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, gt_masks):
        """
        计算主分支和辅助分支的全部损失
        参数:
            outputs: 模型的输出字典
            gt_masks: [bs, H, W]，每个像素是类别 id（从 0 到 num_classes），即 ground truth 掩码
        """
        # 1. 去掉辅助输出，仅保留主分支输出（最后一层）
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        # 2. 将语义分割 mask 转换为目标格式
        targets = self._get_targets(gt_masks)
        # 3. 二分图匹配，返回 indices
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # 4. 统计 batch 中的目标 mask 数量（即前景 query 数）
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        # 5. 多卡训练时做归一化（同步所有卡上的标签数量）
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        # 6. 主分支损失计算
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:  # self.losses : ["labels", "masks"]
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))  # return : {'loss_labels': tensor(...)}, {'loss_ce': tensor(...), 'loss_dice': tensor(...)}
        # 7. 处理辅助输出：aux_outputs 是 decoder 中间层输出
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # outputs["aux_outputs"] : [{"pred_logits": a, "pred_masks": b}, {"pred_logits": a, "pred_masks": b}, ...]
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}  # {'loss_xx_i': tensor(...)}
                    losses.update(l_dict)

        return losses

    # 将语义分割 ground truth mask 转换为训练时需要的 实例级 supervision 格式
    def _get_targets(self, gt_masks):
        targets = []
        for mask in gt_masks:  # 遍历 batch 中每张图片的 gt mask，mask: [H, W]
            # 创建一个新的标签，将255的像素设为 num_classes
            mask_clone = mask.clone()
            ignore_mask = (mask_clone == self.ignore_id)
            mask_clone[ignore_mask] = self.num_classes  # 把255替换成 num_classes（新类别）
            mask = mask_clone
            # [h, w] -> [num_classes, h, w] 语义级掩码转为实例级掩码，每个通道一个类别
            binary_masks = self._get_binary_mask(mask)
            # 提取该图像中实际出现过的类别，例如 [0, 3, 5]，结果默认升序排列
            cls_label = torch.unique(mask)
            # 去掉最后一类即忽略类，只保留前景类别作为目标
            if cls_label[-1] == self.num_classes:
                labels = cls_label[:-1]
            else:
                labels = cls_label
            # 提取对应类别的掩码，即去掉图中没有的类和背景 [num_classes, h, w] -> [num_classes', h, w]
            binary_masks = binary_masks[labels]
            targets.append({
                'masks': binary_masks,  # [num_classes', h, w] 二值掩码，每个类别/实例一个通道
                'labels': labels  # [num_classes'] 每个类别的类别编号，与masks的通道一一对应
            })
        return targets
        
    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class FocalLoss(nn.Module):
    """
    用于语义分割的 Focal Loss
    适配输入:
        pred: [B, num_cls, H, W] (logits, 未经过 softmax)
        target: [B, H, W] (类别id)
    支持 ignore_id 忽略指定标签
    """
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=255, reduction='mean'):
        """
        参数:
            gamma: 聚焦参数, 控制难样本的权重
            alpha: 类别平衡参数 (scalar 或 [num_cls] 的 list/ndarray)
            ignore_index: 需要忽略的标签id
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred: [B, C, H, W] logits
        target: [B, H, W] int64
        """
        # 展平 batch & spatial 维度
        b, c, h, w = pred.shape
        pred = pred.permute(0, 2, 3, 1).reshape(-1, c)   # [B*H*W, C]
        target = target.view(-1)                         # [B*H*W]

        # 忽略 ignore_index
        valid_mask = target != self.ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]

        if target.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        # 计算 log_softmax
        logpt = F.log_softmax(pred, dim=1)                # [N, C]
        pt = torch.exp(logpt)                             # softmax 概率

        # 取出目标类别对应的概率
        logpt = logpt.gather(1, target.unsqueeze(1))      # [N, 1]
        pt = pt.gather(1, target.unsqueeze(1))            # [N, 1]

        # alpha 权重处理
        if isinstance(self.alpha, (float, int)):
            alpha_t = torch.full_like(pt, self.alpha)
        elif isinstance(self.alpha, (list, tuple, torch.Tensor)):
            alpha_t = torch.tensor(self.alpha, device=pred.device)[target]
            alpha_t = alpha_t.unsqueeze(1)
        else:
            raise TypeError("alpha 必须是 float/int 或 list/tuple/tensor")

        # focal loss 公式: -alpha * (1-pt)^gamma * logpt
        loss = -alpha_t * (1 - pt) ** self.gamma * logpt

        # reduction 处理
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class Criterion(object):
    def __init__(self, num_classes, alpha=0.5, gamma=2, weight=None, ignore_index=0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.smooth = 1e-5
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='none')
    
    def get_loss(self, outputs, gt_masks):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             gt_masks: [bs, h_net_output, w_net_output]
        """
        loss_labels = 0.0
        loss_masks = 0.0
        loss_dices = 0.0
        num = gt_masks.shape[0]
        pred_logits = [outputs["pred_logits"].float()] # [bs, num_query, num_classes + 1]
        pred_masks = [outputs['pred_masks'].float()] # [bs, num_query, h, w]
        targets = self._get_targets(gt_masks, pred_logits[0].shape[1], pred_logits[0].device)
        for aux_output in outputs['aux_outputs']:            
            pred_logits.append(aux_output["pred_logits"].float())
            pred_masks.append(aux_output["pred_masks"].float())

        gt_label = targets['labels'] # [bs, num_query]
        gt_mask_list = targets['masks']
        for mask_cls, pred_mask in zip(pred_logits, pred_masks):            
            loss_labels += F.cross_entropy(mask_cls.transpose(1, 2), gt_label)
            # loss_masks += self.focal_loss(pred_result, gt_masks.to(pred_result.device))
            loss_dices += self.dice_loss(pred_mask, gt_mask_list)

        return loss_labels/num, loss_dices/num

    def binary_dice_loss(self, inputs, targets):      
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
        denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.mean()

    def dice_loss(self, predict, targets):    
        bs = predict.shape[0]
        total_loss = 0
        for i in range(bs):
            pred_mask = predict[i]
            tgt_mask = targets[i].to(predict.device)
            dice_loss_value = self.binary_dice_loss(pred_mask, tgt_mask) 
            total_loss += dice_loss_value
        return total_loss/bs

    def focal_loss(self, preds, labels):
        """
        preds: [bs, num_class + 1, h, w]
        labels: [bs, h, w]
        """
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss.mean()

    def _get_binary_mask(self, target):
        y, x = target.size()
        target_onehot = torch.zeros(self.num_classes + 1, y, x)
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
        return target_onehot

    def _get_targets(self, gt_masks, num_query, device):
        binary_masks = []
        gt_labels = []
        for mask in gt_masks:
            mask_onehot = self._get_binary_mask(mask)
            cls_label = torch.unique(mask)
            labels = torch.full((num_query,), 0, dtype=torch.int64, device=gt_masks.device)
            labels[:len(cls_label)] = cls_label           
            binary_masks.append(mask_onehot[cls_label])
            gt_labels.append(labels)
        return {"labels": torch.stack(gt_labels).to(device), "masks": binary_masks}