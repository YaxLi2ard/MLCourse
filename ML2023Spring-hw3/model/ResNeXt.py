import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义 ResNeXt 的 Bottleneck 模块（区别是中间用 group conv）
class ResNeXtBottleneck(nn.Module):
    expansion = 4  # 输出通道扩展倍数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, cardinality=32, base_width=4):
        super(ResNeXtBottleneck, self).__init__()
        D = int(out_channels * (base_width / 64.)) * cardinality  # 计算分组卷积的总通道数

        # 1x1降维卷积
        self.conv1 = nn.Conv2d(in_channels, D, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D)

        # 3x3 分组卷积
        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(D)

        # 1x1升维卷积
        self.conv3 = nn.Conv2d(D, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 用于匹配残差连接维度

    def forward(self, x):
        identity = x  # 残差连接保留原始输入

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 加上残差
        out = self.relu(out)
        return out

# ResNeXt 主体结构
class ResNeXt(nn.Module):
    def __init__(self, block, layers, num_classes=1000, cardinality=32, base_width=4):
        super(ResNeXt, self).__init__()
        self.in_channels = 64  # 初始通道数
        self.cardinality = cardinality
        self.base_width = base_width

        # 初始卷积层：输出64，步长2，7x7卷积
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个阶段，每个包含若干个block
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局池化 + 分类
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * block.expansion, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes)
        )

    # 构造残差层
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # 下采样：匹配维度用于残差连接
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        # 第一个 block 可能需要下采样
        layers.append(block(self.in_channels, out_channels, stride, downsample,
                            cardinality=self.cardinality, base_width=self.base_width))
        self.in_channels = out_channels * block.expansion

        # 后续 block 不需要下采样
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels,
                                cardinality=self.cardinality, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # 初始卷积
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 构造 ResNeXt-50 模型
def resnext50(num_classes=1000):
    return ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], num_classes=num_classes, cardinality=32, base_width=4)

# 测试模型结构
if __name__ == '__main__':
    model = resnext50(num_classes=1000)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)  # 应输出 torch.Size([1, 1000])
