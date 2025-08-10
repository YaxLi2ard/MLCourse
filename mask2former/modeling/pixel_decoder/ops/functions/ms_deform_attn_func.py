# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

# try:
#     import MultiScaleDeformableAttention as MSDA
# except ModuleNotFoundError as e:
#     info_string = (
#         "\n\nPlease compile MultiScaleDeformableAttention CUDA op with the following commands:\n"
#         "\t`cd mask2former/modeling/pixel_decoder/ops`\n"
#         "\t`sh make.sh`\n"
#     )
#     raise ModuleNotFoundError(info_string)


# class MSDeformAttnFunction(Function):
#     @staticmethod
#     def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
#         ctx.im2col_step = im2col_step
#         output = MSDA.ms_deform_attn_forward(
#             value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
#         ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
#         return output

#     @staticmethod
#     @once_differentiable
#     def backward(ctx, grad_output):
#         value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
#         grad_value, grad_sampling_loc, grad_attn_weight = \
#             MSDA.ms_deform_attn_backward(
#                 value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

#         return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    @value 多个特征图展平拼接: bs, sum(h, w), num_head, dim
    @sampling_locations 采样点位置坐标: bs, sum(h, w), num_head, num_layer, 4, 2
    @attention_weights 采样点权重: bs, sum(h, w), num_head, num_layer, 4
    """
    N_, S_, M_, Dim = value.shape  # N_ batchsize, S_ 所有特征点数量, M_ 头数量, Dim 每个头的特征长度
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape  # _ batchsize, Lq_ 所有特征点数量, M_ 头数, L_ 层次数, P_ 点数量, _ 点坐标长度为2（x，y）
    ''' 将 拼接起来的 flatten 的 value 切分回每一层尺度特征图 value_list[i]: [bs, Hi*Wi, M, Dim] '''
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1 # 把范围从[0,1]转换到[-1,1], F.grid_sample要求grid的范围是[-1,1]
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, H_, W_
        ''' 取出该层的特征图并转换 [B, Hi*Wi, M, D] -> [B, Hi*Wi, M*D] -> [B, M*D, Hi*Wi] -> [B*M, D, Hi, Wi] '''
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, Dim, H_, W_) # eg. [bs * 8, 32, 28, 28, 28]
        # N_, Lq_, M_, P_, 3 -> N_, M_, Lq_, P_, 3 -> N_*M_, Lq_, P_, 3
        sampling_grid_l_ = sampling_grids[:, :, :, lid_]  # [B, H*W, M, P, 2]
        ''' 取出所有点对该层的采样点并转换 [B, S(H*W), M, P, 2] -> [B, M, S(H*W), P, 2] -> [B*M, S(H*W), P, 2] '''
        sampling_grid_l_ = sampling_grid_l_.transpose(1, 2).flatten(0, 1) # eg. [bs * 8, 1045, 3, 3]
        # N_*M_, D_, Lq_, P_
        ''' 采样 [B*M, D, S(H*W), P] 这代表所有点在lid_层的采样点特征'''
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False) # eg. [bs * 8, 32, 1045, 4]
        sampling_value_list.append(sampling_value_l_)

    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    ''' 采样点权重转换 [B, S(H*W), M, L, P] -> [B, M, S(H*W), L, P] -> [B*M, 1, S(H*W), L*P] dim1的1代表所有通道权重相同'''
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_) # eg. [bs * 8, 1, 1045, 4 * 4], 4个特征层 * 4个采样点
    # torch.stack(sampling_value_list, dim=-2): [bs * 8, 32, 1045, 4, num_layer] -> [bs * 8, 32, 1045, 4 * 4], 4个特征层 * 4个采样点
    '''
    stack -> [B*M, D, S(H*W), L, P] -> flatten -> [B*M, D, S(H*W), L*P] 【代表每个样本，每个头，每个特征点要采样的L*P个特征点向量，L个层级，每个层级采样P个点】
    -> * attention_weights [B*M, 1, S(H*W), L*P] 【代表每个样本，每个头，每个特征点要采样的L*P个特征点向量的权重】 -> [B*M, D, S(H*W), L*P]
    -> sum(-1) -> [B*M, D, S(H*W)] 【代表每个样本，每个头，每个特征点要采样的L*P个特征点向量的加权和】
    -> view(N_, M_*Dim, Lq_) -> [B, M*D, S(H*W)] 相当于把多个头拼接起来
    -> transpose(1, 2) -> [B, S(H*W), M*D] 换一下顺序
    看以看出操作进行时，头和batch一个层级，对于每个头，聚合所有层级的采样点，然后多个头拼接，多个头之间不参与加权和，加权和的是多个层级的多个采样点
    '''
    output = (torch.stack(sampling_value_list, dim=-2).squeeze(2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*Dim, Lq_)
    return output.transpose(1, 2).contiguous()
