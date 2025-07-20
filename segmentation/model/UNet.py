import torch
import torch.nn as nn
import torch.nn.functional as F
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

# 下采样模块：双卷积 + 最大池化
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x_down = self.pool(x)
        return x, x_down  # 返回特征和池化后的结果，用于后续跳跃连接

# 上采样模块：上采样 + 拼接 + 双卷积
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)  # 拼接后通道数翻倍

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print("x1 shape", x1.shape)
        # print("x2 shape", x2.shape)
        # 保证拼接时尺寸一致（处理尺寸不整除问题）
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 最终输出层
class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# UNet 主体
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_c=64):
        super().__init__()
        self.down1 = Down(in_channels, base_c*1)
        self.down2 = Down(base_c*1, base_c*2)
        self.down3 = Down(base_c*2, base_c*4)
        self.down4 = Down(base_c*4, base_c*8)
        self.midc = DoubleConv(base_c*8, base_c*16)
        self.up1 = Up(base_c*16, base_c*8)
        self.up2 = Up(base_c*8, base_c*4)
        self.up3 = Up(base_c*4, base_c*2)
        self.up4 = Up(base_c*2, base_c*1)
        self.outc = OutConv(base_c, num_classes)

    def forward(self, x):
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        x = self.midc(x)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

def safe_check(name, tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"❌ {name} has NaN or Inf!")
    else:
        print(f"✅ {name} ok: [{tensor.min().item()}, {tensor.max().item()}]")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(num_classes=21, base_c=64)
    model.to(device)
    # x = torch.randn(1, 3, 320, 320)
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