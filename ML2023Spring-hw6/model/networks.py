# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# =====================
# StyleGAN2-ADA 网络结构实现
# 本文件定义了生成器、判别器及其子模块的PyTorch实现。
#
# =====================

import numpy as np
import torch
from model.torch_utils import misc
from model.torch_utils import persistence
from model.torch_utils.ops import conv2d_resample
from model.torch_utils.ops import upfirdn2d
from model.torch_utils.ops import bias_act
from model.torch_utils.ops import fma

#----------------------------------------------------------------------------

@misc.profiled_function
# 二范数归一化，常用于特征向量的归一化处理，防止数值爆炸。
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@misc.profiled_function
# 调制卷积（Modulated Conv2d），StyleGAN的核心创新之一。
# 支持风格调制、去调制、上采样、下采样、噪声注入等。
def modulated_conv2d(
    x,                          # 输入特征图 [batch_size, in_channels, in_height, in_width]
    weight,                     # 卷积核权重 [out_channels, in_channels, kernel_height, kernel_width]
    styles,                     # 风格调制系数 [batch_size, in_channels]
    noise           = None,     # 可选噪声张量
    up              = 1,        # 上采样倍数
    down            = 1,        # 下采样倍数
    padding         = 0,        # 填充大小
    resample_filter = None,     # 重采样时的低通滤波器
    demodulate      = True,     # 是否去调制（归一化权重）
    flip_weight     = True,     # 是否翻转权重（区分卷积/相关操作）
    fused_modconv   = True,     # 是否融合为单一高效操作
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # FP16下预归一化，防止溢出
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # 计算每个样本的权重和去调制系数
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # 非融合模式：分步执行调制、卷积、去调制、加噪声
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # 融合模式：分组卷积实现高效批量操作
    with misc.suppress_tracer_warnings(): # 该值会被视为常量
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
# 全连接层，支持自定义激活函数和学习率缩放。
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # 输入特征数
        out_features,               # 输出特征数
        bias            = True,     # 是否加偏置
        activation      = 'linear', # 激活函数类型
        lr_multiplier   = 1,        # 学习率缩放因子
        bias_init       = 0,        # 偏置初始化
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
# 二维卷积层，支持上采样、下采样、激活函数、权重/偏置可训练等功能。
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # 输入通道数
        out_channels,                   # 输出通道数
        kernel_size,                    # 卷积核尺寸
        bias            = True,         # 是否加偏置
        activation      = 'linear',     # 激活函数类型
        up              = 1,            # 上采样倍数
        down            = 1,            # 下采样倍数
        resample_filter = [1,3,3,1],    # 重采样时的低通滤波器
        conv_clamp      = None,         # 输出裁剪范围
        channels_last   = False,        # 是否使用channels_last内存格式
        trainable       = True,         # 权重是否可训练
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        # 权重归一化
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # 仅上采样为1时加速
        # 卷积+重采样
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        # 激活函数与裁剪
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
# 映射网络，将输入潜变量z和条件c映射到W空间，支持标签嵌入、归一化、截断等。
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # 输入潜变量z的维度
        c_dim,                      # 条件标签c的维度
        w_dim,                      # 中间潜变量w的维度
        num_ws,                     # 输出w的数量（广播用）
        num_layers      = 8,        # 映射层数
        embed_features  = None,     # 标签嵌入维度
        layer_features  = None,     # 每层特征数
        activation      = 'lrelu',  # 激活函数
        lr_multiplier   = 0.01,     # 学习率缩放
        w_avg_beta      = 0.995,    # w均值滑动平均系数
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # 嵌入、归一化、拼接z和c
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # 多层映射
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # 更新w的滑动平均
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # 广播
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # 截断技巧
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
# 合成层，StyleGAN2生成器的基本构建块，支持风格调制、噪声注入、激活等。
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # 输入通道数
        out_channels,                   # 输出通道数
        w_dim,                          # W空间维度
        resolution,                     # 当前层分辨率
        kernel_size     = 3,            # 卷积核尺寸
        up              = 1,            # 上采样倍数
        use_noise       = True,         # 是否加噪声
        activation      = 'lrelu',      # 激活函数
        resample_filter = [1,3,3,1],    # 重采样低通滤波器
        conv_clamp      = None,         # 输出裁剪
        channels_last   = False,        # 是否channels_last格式
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        # x: 输入特征图，w: 风格向量
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution])
        styles = self.affine(w)

        # 噪声生成
        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # 仅上采样为1时加速
        # 调制卷积
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        # 激活与裁剪
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
# ToRGB层，将特征图转换为RGB图像（或多通道输出），用于生成器每个分辨率的输出。
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        # x: 特征图，w: 风格向量
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
# 合成块，生成器的分辨率层级模块，包含多个SynthesisLayer和ToRGBLayer。
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # 输入通道数，0表示首层
        out_channels,                       # 输出通道数
        w_dim,                              # W空间维度
        resolution,                         # 当前块分辨率
        img_channels,                       # 输出图像通道数
        is_last,                            # 是否为最后一块
        architecture        = 'skip',       # 结构类型
        resample_filter     = [1,3,3,1],    # 重采样低通滤波器
        conv_clamp          = None,         # 输出裁剪
        use_fp16            = False,        # 是否用FP16
        fp16_channels_last  = False,        # 是否channels_last格式
        **layer_kwargs,                     # 传递给SynthesisLayer的参数
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            # 首层为可学习常量
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            # 上采样+卷积
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        # 主卷积层
        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        # ToRGB层
        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        # ResNet结构的skip分支
        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, **layer_kwargs):
        # ws: [batch, num_conv+num_torgb, w_dim]
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # 该值会被视为常量
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # 输入
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # 主体卷积
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB输出
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
# 合成网络，StyleGAN2生成器的主干，按分辨率堆叠多个SynthesisBlock。
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # W空间维度
        img_resolution,             # 输出分辨率
        img_channels,               # 输出通道数
        channel_base    = 32768,    # 通道数基数
        channel_max     = 512,      # 单层最大通道数
        num_fp16_res    = 0,        # 用FP16的高分辨率层数
        **block_kwargs,             # 传递给SynthesisBlock的参数
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        # ws: [batch, num_ws, w_dim]
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
# 生成器主类，包含映射网络和合成网络，forward时先映射z/c，再合成图像。
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # 输入潜变量z的维度
        c_dim,                      # 条件标签c的维度
        w_dim,                      # W空间维度
        img_resolution,             # 输出分辨率
        img_channels,               # 输出通道数
        mapping_kwargs      = {},   # MappingNetwork参数
        synthesis_kwargs    = {},   # SynthesisNetwork参数
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        # 先映射z/c到w，再合成图像
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, **synthesis_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
# 判别器的分辨率块，支持ResNet结构、FP16、skip连接等。
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # 输入通道数，0为首层
        tmp_channels,                       # 中间通道数
        out_channels,                       # 输出通道数
        resolution,                         # 当前块分辨率
        img_channels,                       # 输入图像通道数
        first_layer_idx,                    # 当前块的首层索引
        architecture        = 'resnet',     # 结构类型
        activation          = 'lrelu',      # 激活函数
        resample_filter     = [1,3,3,1],    # 重采样低通滤波器
        conv_clamp          = None,         # 输出裁剪
        use_fp16            = False,        # 是否用FP16
        fp16_channels_last  = False,        # 是否channels_last格式
        freeze_layers       = 0,            # 冻结前几层
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            # 首层或skip结构的fromRGB
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # 输入
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # fromRGB
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # 主体卷积
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
# Minibatch标准差层，用于提升判别器对模式崩溃的鲁棒性。
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        # 计算小批量内的标准差特征，并拼接到输入通道
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] 分组
        y = y - y.mean(dim=0)               # 减去组内均值
        y = y.square().mean(dim=0)          # 组内方差
        y = (y + 1e-8).sqrt()               # 组内标准差
        y = y.mean(dim=[2,3,4])             # 通道和空间均值
        y = y.reshape(-1, F, 1, 1)          # 补齐维度
        y = y.repeat(G, 1, H, W)            # 扩展到全图
        x = torch.cat([x, y], dim=1)        # 拼接到输入
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
# 判别器尾部，包含MinibatchStdLayer、全连接层、条件投影等。
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # 输入通道数
        cmap_dim,                       # 条件映射维度
        resolution,                     # 当前分辨率
        img_channels,                   # 输入图像通道数
        architecture        = 'resnet', # 结构类型
        mbstd_group_size    = 4,        # MinibatchStd分组大小
        mbstd_num_channels  = 1,        # MinibatchStd通道数
        activation          = 'lrelu',  # 激活函数
        conv_clamp          = None,     # 输出裁剪
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        # x: 特征图，img: 输入图像，cmap: 条件映射
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # 未用
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # fromRGB
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # 主体
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # 条件投影
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
# 判别器主类，按分辨率堆叠多个DiscriminatorBlock，最后接Epilogue。
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # 条件标签c的维度
        img_resolution,                 # 输入分辨率
        img_channels,                   # 输入通道数
        architecture        = 'resnet', # 结构类型
        channel_base        = 32768,    # 通道数基数
        channel_max         = 512,      # 单层最大通道数
        num_fp16_res        = 0,        # 用FP16的高分辨率层数
        conv_clamp          = None,     # 输出裁剪
        cmap_dim            = None,     # 条件映射维度
        block_kwargs        = {},       # DiscriminatorBlock参数
        mapping_kwargs      = {},       # MappingNetwork参数
        epilogue_kwargs     = {},       # DiscriminatorEpilogue参数
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, **block_kwargs):
        # img: 输入图像，c: 条件标签
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

#----------------------------------------------------------------------------
