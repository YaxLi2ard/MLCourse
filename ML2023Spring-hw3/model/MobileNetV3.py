import torch
import torch.nn as nn
import torch.nn.functional as F


# 轻量级注意力模块：SE（Squeeze-and-Excitation）模块
class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
        )
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc(out)
        out = self.hardsigmoid(out)
        return x * out  # 通道注意力


# MobileNetV3 中的基础模块：包含卷积、BN、激活、SE、残差连接
class MobileBottleneck(nn.Module):
    def __init__(self, in_dim, exp_dim, out_dim, kernel_size, use_se, activation, stride):
        super(MobileBottleneck, self).__init__()
        # 只有输入和输出形状相同时才有残差连接
        self.use_res_connect = (stride == 1 and in_dim == out_dim)

        layers = []

        # 1x1 升维
        if in_dim != exp_dim:  # 第一个bottleneck输入通道与升维通道数一样，不使用1*1卷积升维
            layers.append(nn.Conv2d(in_dim, exp_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(exp_dim))
            layers.append(activation())

        # 3x3 DW 卷积
        layers.append(nn.Conv2d(exp_dim, exp_dim, kernel_size=kernel_size,
                                stride=stride, padding=kernel_size // 2, groups=exp_dim, bias=False))
        layers.append(nn.BatchNorm2d(exp_dim))
        layers.append(activation())

        # SE模块（可选）
        if use_se:
            layers.append(SEModule(exp_dim))

        # 1x1 降维
        layers.append(nn.Conv2d(exp_dim, out_dim, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res_connect:
            return x + out  # 残差连接
        else:
            return out


# MobileNetV3-Large 主体网络
class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3Large, self).__init__()

        # 激活函数选择
        def hswish(): return nn.Hardswish()
        def relu(): return nn.ReLU()

        # Stem 卷积
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )

        # 主体卷积结构（根据论文中 MobileNetV3-Large 的配置）
        self.blocks = nn.Sequential(
            # in_channels, out_channels, kernel_size, stride, expansion, use_se, activation
            # in_dim, exp_dim, out_dim, kernel_size, use_se, activation, stride
            MobileBottleneck(16, 16, 16, 3, False, relu, 1),
            MobileBottleneck(16, 64, 24, 3, False, relu, 2),
            MobileBottleneck(24, 72, 24, 3, False, relu, 1),
            MobileBottleneck(24, 72, 40, 5, True, relu, 2),
            MobileBottleneck(40, 120, 40, 5, True, relu, 1),
            MobileBottleneck(40, 120, 40, 5, True, relu, 1),
            MobileBottleneck(40, 240, 80, 3, False, hswish, 2),
            MobileBottleneck(80, 200, 80, 3, False, hswish, 1),
            MobileBottleneck(80, 184, 80, 3, False, hswish, 1),
            MobileBottleneck(80, 184, 80, 3, False, hswish, 1),
            MobileBottleneck(80, 480, 112, 3, True, hswish, 1),
            MobileBottleneck(112, 672, 112, 3, True, hswish, 1),
            MobileBottleneck(112, 672, 160, 5, True, hswish, 2),
            MobileBottleneck(160, 960, 160, 5, True, hswish, 1),
            MobileBottleneck(160, 960, 160, 5, True, hswish, 1),
        )

        # 最后阶段
        self.final = nn.Sequential(
            nn.Conv2d(160, 960, kernel_size=1, bias=False),
            nn.BatchNorm2d(960),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(960, 1280, kernel_size=1),
            nn.Hardswish(),
            nn.Conv2d(1280, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final(x)
        return x.view(x.size(0), -1)  # 展平为 [B, num_classes]

# 测试
if __name__ == '__main__':
    model = MobileNetV3Large(num_classes=11)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)  # 应为 [1, 11]