import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # 全连接层用于压缩通道数
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)  # [B, C]
        max_pool = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)  # [B, C]

        # MLP处理
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        # 合并并激活（通道注意力）
        attn = torch.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        return x * attn  # 广播乘法，通道注意力作用


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 卷积层用于空间注意力，保持输入输出通道为1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.last_attn = None  # 用于保存注意力权重

    def forward(self, x):
        # x: [B, C, H, W]
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # 拼接后卷积
        pool = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, H, W]
        attn = torch.sigmoid(self.conv(pool))  # [B, 1, H, W]
        self.last_attn = attn.detach()
        return x * attn  # 空间注意力作用


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        res = x
        # 先计算通道注意力，再计算空间注意力
        x = self.channel_att(x)
        x = self.spatial_att(x)
        x = x + res
        return x


class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.3):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.last_attn_weights = None  # 用于保存注意力权重

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        # cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, C]
        cls_tokens = torch.mean(x, dim=1, keepdim=True)  # [B, 1, C]
        x = torch.cat([cls_tokens, x], dim=1)          # [B, 1+H*W, C]
        
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)
        self.last_attn_weights = attn_weights.detach()  # 保存注意力

        x = x + attn_out  # [B, 1+H*W, C]
        cls_out = x[:, 0]   # 取出 CLS token 输出 [B, C]
        return cls_out


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

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, attention=False):
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

        self.attention = attention
        if self.attention:
            self.addition = CBAM(out_channels * self.expansion, reduction_ratio=16, spatial_kernel_size=7)
        
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

        if self.attention:
            out = self.addition(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

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
        self.layer1 = self._make_layer(block, 64, layers[0], attention=False)   # 64维输出，重复 layers[0] 次
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, attention=False)  # 输出通道128，步长2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, attention=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, attention=False)

        # 全局平均池化 + 全连接输出
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化为 1x1

        self.sa = SelfAttention(d_model=512 * block.expansion, nhead=8, dropout=0.3)
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * block.expansion, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes)            
        )

    # 构建残差层
    def _make_layer(self, block, out_channels, blocks, stride=1, attention=False):
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
        layers.append(block(self.in_channels, out_channels, stride, downsample, attention=False))
        self.in_channels = out_channels * block.expansion
        # 后续blocks直接连接，不需要下采样downsample
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, attention=False))
        if attention:
            cbam = CBAM(out_channels * block.expansion, reduction_ratio=16, spatial_kernel_size=7)
            layers.append(cbam)
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

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)

        x = self.sa(x)
        
        x = self.fc(x)
        return x


# # 构建ResNet18模型（使用 BasicBlock）
# def resnet18(num_classes=1000):
#     return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

# # 构建ResNet34模型（使用 BasicBlock）
# def resnet34(num_classes=1000):
#     return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

# 构建ResNet50模型（使用 Bottleneck）
def resnet50_a(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


if __name__ == '__main__':
    model = resnet50(num_classes=1000)
    x = torch.randn(3, 3, 256, 256)
    y = model(x)
    print(y.shape)