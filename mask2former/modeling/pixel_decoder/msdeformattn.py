#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   msdeformattn.py
@Time    :   2022/10/02 16:51:09
@Author  :   BQH 
@Version :   1.0
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   修改自Mask2former,移除detectron2依赖
'''

# here put the import lib

import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F


from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from ..transformer_decoder.transformer import _get_clones, _get_activation_fn
from .ops.modules import MSDeformAttn

# MSDeformAttn Transformer encoder in deformable detr
''' 编码器中的一个 Transformer 层 可变形注意力+ffn '''
class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

''' 堆叠多个编码器层的整体编码器模块 '''
class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    ''' 为不同的特征层生成归一化后的 2D 网格参考点 抽象'''
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 生成一个 cell 中心点的网格坐标，以 .5 开始和结束
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device), indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)  # [1, H_ * W_, 2]
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        ''' 不太理解为什么前面除上了valid_ratios，这里又乘了valid_ratios '''
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        ''' 一张特征图可以看成多个网格，某个位置（网格）的参考点其实就是这个网格的中心点的坐标（0-1归一化），它在不同特征层的参考点都是这个 '''
        ''' 输出形状[1, sum(h*w), 1, 2] 第一维是batch维度，第三维是n_levels，同一个位置对不同尺寸层的参考点相同，所以用1代替 '''
        return reference_points  # [1, sum(h*w), 1, 2]

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output

''' 对多个特征图（来自 backbone）进行预处理（展平并拼接）并送入 Transformer 编码器 '''
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,
        ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        # 遍历不同尺度的 特征图、mask（全有效）、位置编码
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # 展平 [b,c,h,w] -> [b,c,h*w] -> [b,h*w,c]
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            # 展平 [b,c,h,w] -> [b,c,h*w] -> [b,h*w,c]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            # 层次编码+位置编码
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        return memory, spatial_shapes, level_start_index


''' 论文里的 PixelDecoder 实现 (MSDeformAttnTransformer + FFN) '''
class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(
        self,
        input_shape,  # eg. {"res2": {channel:256, stride:4}, "res3": {512, 8}, "res4": {1024, 16}, "res5": {2048, 32}}
        transformer_dropout=0.1,
        transformer_nheads=8,
        transformer_dim_feedforward=2048,
        transformer_enc_layers=6,
        conv_dim=256,
        mask_dim=256,

        # deformable transformer encoder args
        transformer_in_features= ["res3", "res4", "res5"],
        common_stride=4,
    ):
        super().__init__()
        # backbone中["res3", "res4", "res5"]特征层的(channel, stride), eg. [(32,4), (64, 8), (128, 16), (256, 32)]
        transformer_input_shape = {k: v for k, v in input_shape.items() if k in transformer_in_features} 
        
        # this is the input shape of pixel decoder        
        self.in_features = [k for k, v in input_shape.items()]  # starting from "res3" to "res5"        
        self.feature_channels = [v.channel for k, v in input_shape.items()] # eg. [16, 64, 128, 256]
        
        # this is the input shape of transformer encoder (could use less features than pixel decoder        
        self.transformer_in_features = [k for k, v in transformer_input_shape.items()]  # starting from "res3" to "res5"
        transformer_in_channels = [v.channel for k, v in transformer_input_shape.items()] # eg. [64, 128, 256]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape.items()]  # to decide extra FPN layers

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res3)
            ''' 对不同尺度的特征图进行一个1*1卷积变换，使得不同尺寸特征图的通道数一致 '''
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        ''' 前面的encoder 使用可变形注意力，使得特征图每个位置交互不同尺度特征图上的、相近位置上的信息 '''
        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = nn.Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_features)

        # 使用 3 个特征层级参与 decoder
        self.maskformer_num_feature_levels = 3  # always use 3 scales
        # mask 预测分支对齐到的分辨率（如 4，对应 1/4 原图）
        self.common_stride = common_stride

        # extra fpn levels
        # 获取 transformer 输出中，最小stride（最大分辨率）
        stride = min(self.transformer_feature_strides)
        ''' 
        计算FPN层数，假设特征中下采样倍数（stride）为 2、4、8、16、32，encoder采取[16,32]尺度为输入（输出）
        那么stride = min(self.transformer_feature_strides) = 16，假设common_stride为2，那么需要log(16)-log(2)=3个FPN层
        即common_stride为FPN左边（Bottom-up）最下边的分辨率，encoder输出中最大的分辨率代表FPN右边（Top-down）最上边的分辨率
        
        可以看成是有一系列从大到小的特征图，假设ommon_stride是下标为i的特征图的stride，encoder采取下标j以后的特征图为输出
        其中j>i，stride_j>stride_i，size_j<size_i，因此现在 f_j 要不断上采样并与 f_(j-1) - f_i 融合，就是FPN
        在mask2former里encoder的输出最小stride是8，仅需上采样、融合1次
        '''
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        # 构建用于 FPN 的横向连接模块（adapter）和输出模块（layer）
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]): # res2 -> fpn
            # 横向卷积模块：1x1卷积 + GN + ReLU，用于将 backbone 的特征图通道数转为 conv_dim
            lateral_conv = nn.Sequential(nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                                         nn.GroupNorm(32, conv_dim),
                                         nn.ReLU(inplace=True))
            # 输出卷积模块：3x3卷积 + GN + ReLU，用于进一步处理融合后的特征
            output_conv = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, kernel_size=3,  stride=1,  padding=1),
                                        nn.GroupNorm(32, conv_dim),
                                        nn.ReLU(inplace=True))
            
            weight_init.c2_xavier_fill(lateral_conv[0])
            weight_init.c2_xavier_fill(output_conv[0])
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    def forward_features(self, features):
        srcs = []
        pos = []
        # Reverse feature maps into top-down order (from low to high resolution), 'res5' -> 'res3'
        # 不同尺度特征图变换为相同通道
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision [b,c,h,w]
            srcs.append(self.input_proj[idx](x))  # [b, c', h, w]
            pos.append(self.pe_layer(x))  # [b, c', h, w]

        # 可变形注意力 transformer encode
        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        # 把拼接起来的多尺度特征图分割回去
        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        # 把多个特征图reshape回原来的形状 [b, c, h, w]
        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)
            out.append(y)

        # 从 out 中选取前 maskformer_num_feature_levels 个特征，就是encoder的原生输出
        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        ''' 
        out[-1]为FPN的输出，mask_features(out[-1])对其进行变换
        out[0]为encoder的输出，且是最小尺度特征图
        multi_scale_features是多个encoder的输出
        '''
        return self.mask_features(out[-1]), out[0], multi_scale_features