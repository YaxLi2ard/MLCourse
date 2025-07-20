import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, parameter_count_table

class FCN(nn.Module):
    def __init__(self, num_classes=21, backbone='resnet50', pretrained=True):
        super(FCN, self).__init__()

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
        self.stage1 = nn.Sequential(
            resnet.conv1,  # [B, 64, H/2, W/2]
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # [B, 64, H/4, W/4]
            resnet.layer1  # [B, 256, H/4, W/4]
        )
        self.stage2 = resnet.layer2  # [B, 512, H/8, W/8]
        self.stage3 = resnet.layer3  # [B, 1024, H/16, W/16]
        self.stage4 = resnet.layer4  # [B, 2048, H/32, W/32]

        # 1x1 卷积代替 FC，分类为每像素预测 num_classes
        self.head = nn.Conv2d(2048, num_classes, kernel_size=1)

        # 用于 skip 连接（可选）
        self.skip1 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.skip2 = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]  # 原始尺寸 (H, W)
        # backbone encode
        x1 = self.stage1(x)  # [B, 256, H/4, W/4]
        x2 = self.stage2(x1)  # [B, 512, H/8, W/8]
        x3 = self.stage3(x2)  # [B, 1024, H/16, W/16]
        x4 = self.stage4(x3)  # [B, 2048, H/32, W/32]
        # decode
        score = self.head(x4)  # [B, C, H/32, W/32]
        # 输出上采样后与layer3输出连接 32->16 + 16
        score = F.interpolate(score, size=x3.shape[2:], mode='bilinear', align_corners=False)
        score += self.skip1(x3)
        # 输出上采样后与layer2输出连接 16->8 + 8
        score = F.interpolate(score, size=x2.shape[2:], mode='bilinear', align_corners=False)
        score += self.skip2(x2)
        # 最终上采样至输入图像大小 8->1
        out = F.interpolate(score, size=input_size, mode='bilinear', align_corners=False)
        return out  # [B, num_classes, H, W]

if __name__ == '__main__':
    model = FCN(num_classes=21, backbone='resnet50', pretrained=True)
    # x = torch.randn(1, 3, 321, 311)
    # out = model(x)
    # print(out.shape)

    x = torch.randn(1, 3, 310, 310)
    inputs = x
    model.eval()
    flops = FlopCountAnalysis(model, inputs)
    print(f"Total FLOPs: {(flops.total() / 1e9):.3f} G")