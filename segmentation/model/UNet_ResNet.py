import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from fvcore.nn import FlopCountAnalysis, parameter_count_table

# 基础模块：双卷积（Conv -> ReLU -> Conv -> ReLU）
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# 上采样模块：上采样 + 拼接 + 双卷积
class Up(nn.Module):
    def __init__(self, x1_dim, x2_dim, out_dim):
        super().__init__()
        x1_mid_dim = x1_dim // 2
        self.up = nn.ConvTranspose2d(x1_dim, x1_mid_dim, kernel_size=2, stride=2)
        self.conv = DoubleConv(x1_mid_dim + x2_dim, out_dim)  # 拼接后通道数增加

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print("x1 shape", x1.shape)
        # print("x2 shape", x2.shape)
        # 保证拼接时尺寸一致（处理尺寸不整除问题）
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        if x2 is not None:
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        return self.conv(x)

# 最终输出层
class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# UNet 主体
class UNet_ResNet(nn.Module):
    def __init__(self, num_classes=1, backbone='resnet50', pretrained=True):
        super().__init__()
        # 使用torchvision官方的 ResNet 作为 encoder
        if backbone == 'resnet50':
            resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == 'resnet34':
            resnet = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == 'resnet18':
            resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError('Unsupported backbone')
            
        # 提取中间特征用于 skip connection
        self.stage0 = nn.Sequential(
            resnet.conv1,  # [B, 64, H/2, W/2]
            resnet.bn1,
            resnet.relu,  # [B, 64, H/2, W/2]
        )
        self.stage1 = nn.Sequential(
            resnet.maxpool,  # [B, 64, H/4, W/4]
            resnet.layer1  # [B, 256, H/4, W/4]
        )
        self.stage2 = resnet.layer2  # [B, 512, H/8, W/8]
        self.stage3 = resnet.layer3  # [B, 1024, H/16, W/16]
        self.stage4 = resnet.layer4  # [B, 2048, H/32, W/32]

        base_c = 64
        self.dp = nn.Dropout(p=0.1)
        self.midc = DoubleConv(base_c*8, base_c*16)
        self.up1 = Up(base_c*32, base_c*16, base_c*16)
        self.up2 = Up(base_c*16, base_c*8, base_c*8)
        self.up3 = Up(base_c*8, base_c*4, base_c*4)
        self.up4 = Up(base_c*4, base_c*1, base_c*2)
        self.up5 = Up(base_c*2, 0, base_c)
        self.outc = OutConv(base_c, num_classes)

    def forward(self, x):
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        x = self.up1(x4, self.dp(x3))
        x = self.up2(x, self.dp(x2))
        x = self.up3(x, self.dp(x1))
        x = self.up4(x, self.dp(x0))
        x = self.up5(x, None)
        logits = self.outc(x)
        return logits

def safe_check(name, tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"❌ {name} has NaN or Inf!")
    else:
        print(f"✅ {name} ok: [{tensor.min().item()}, {tensor.max().item()}]")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet_ResNet(num_classes=21, backbone='resnet50', pretrained=True)
    model.to(device)
    
    # x = torch.randn(1, 3, 320, 320).to(device)
    # y = model(x)
    # print(y.shape)

    x = torch.randn(1, 3, 320, 320).to(device)
    inputs = x
    model.eval()
    flops = FlopCountAnalysis(model, inputs)
    print(f"Total FLOPs: {(flops.total() / 1e9):.3f} G")
    # 参数量统计
    # print("Parameter Count:")
    # print(parameter_count_table(model))