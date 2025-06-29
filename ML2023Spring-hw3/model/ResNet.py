import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义基本的残差模块（BasicBlock），用于 ResNet-18 / ResNet-34
class BasicBlock(nn.Module):
    expansion = 1  # 扩展倍数（输出通道倍数），basicblock里没用，用于兼容 Bottleneck 结构， resnet类中会查看该变量，检查输入输出通道是否相等

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 如果维度不同或步长不为1，需要下采样以匹配尺寸

    def forward(self, x):
        identity = x  # 原始输入用于跳连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # 维度不一致时调整

        out += identity  # 残差连接
        out = self.relu(out)
        return out

# 定义Bottleneck模块（用于ResNet-50 / 101 / 152）
class Bottleneck(nn.Module):
    expansion = 4  # 输出通道扩展倍数：最终输出是中间维度的4倍

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1降维
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3x3卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1x1升维
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x

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

        out += identity
        out = self.relu(out)
        return out

# ResNet 主体结构
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64  # 初始通道数

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个残差层，每个 layer 包含若干 block
        self.layer1 = self._make_layer(block, 64, layers[0])   # 64维输出，重复 layers[0] 次
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 输出通道128，步长2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化 + 全连接输出
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化为 1x1

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * block.expansion, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes)            
        )

    # 构建残差层
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # 如果输入通道数不等于输出通道数，或步长不为1，需要下采样
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        # 第一个block可能需要下采样downsample
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        # 后续blocks直接连接，不需要下采样downsample
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # 初始卷积
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 四个stage
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 构建ResNet18模型（使用 BasicBlock）
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

# 构建ResNet34模型（使用 BasicBlock）
def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

# 构建ResNet50模型（使用 Bottleneck）
def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


if __name__ == '__main__':
    model = resnet50(num_classes=1000)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)