import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table

# -----------------------------------
# 空洞空间金字塔池化模块 ASPP
# -----------------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[1, 5, 10, 15]):
        super(ASPP, self).__init__()

        self.blocks = nn.ModuleList()
        # self.blocks.append(
        #     nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, 1, bias=False),
        #         nn.BatchNorm2d(out_channels),
        #         nn.ReLU(inplace=True)
        #     )
        # )
        for rate in rates:
            if rate == 1:
                self.blocks.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                self.blocks.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                )

        # 全局平均池化分支
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 合并后的1x1卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(rates) * out_channels + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[2:]  # 高宽

        aspp_outs = [block(x) for block in self.blocks]

        # 全局池化输出需要上采样
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=False)

        # 拼接所有分支
        x = torch.cat(aspp_outs + [global_feat], dim=1)
        return self.project(x)


# -----------------------------------
# DeepLabV3 Head（ASPP + 1x1 conv）
# -----------------------------------
class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__()
        self.aspp = ASPP(in_channels, 256)
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        x = self.aspp(x)
        x = self.classifier(x)
        return x

# -----------------------------------
# BackBone 调整resnet
# -----------------------------------
class BackBone(nn.Module):
    def __init__(self, backbone):
        super(BackBone, self).__init__()
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        # 修改layer3和layer4为stride=1，调整dilation
        self.layer3 = self._modify_resnet_layer(backbone.layer3, dilate=[1, 2, 2, 2, 2, 2])
        self.layer4 = self._modify_resnet_layer(backbone.layer4, dilate=[2, 4, 4])

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        mid_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, mid_feat

    def _modify_resnet_layer(self, layer, dilate):
        """
        将layer中所有block的stride改为1，并根据dilate调整conv2的dilation和padding
        """
        for i, block in enumerate(layer):
            # 修改第一个block的下采样为stride=1
            if i == 0:
                if hasattr(block, 'downsample') and block.downsample is not None:
                    block.downsample[0].stride = (1, 1)
                block.conv2.stride = (1, 1)
            # 修改空洞卷积（conv2）
            block.conv2.dilation = (dilate[i], dilate[i])
            block.conv2.padding = (dilate[i], dilate[i])
        return layer


class Decoder(nn.Module):
    def __init__(self, low_level_dim, x_dim, num_classes):
        super(Decoder, self).__init__()
        mid_dim = 53
        self.conv1 = nn.Conv2d(low_level_dim, mid_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_dim)
        self.relu = nn.ReLU(inplace=True)

        self.output = nn.Sequential(
            nn.Conv2d(mid_dim + x_dim, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1, stride=1),
        )

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x

# -----------------------------------
# DeepLabV3（完整结构）
# -----------------------------------
class DeepLabV3p(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', pretrained=True):
        super(DeepLabV3p, self).__init__()

        # 1. 主干网络：ResNet50
        if backbone == 'resnet50':
            resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == 'resnet101':
            resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 去掉最后的 avgpool 和 fc
        # backbone = nn.Sequential(
        #     resnet.conv1,  # [B, 64, H/2, W/2]
        #     resnet.bn1,
        #     resnet.relu,
        #     resnet.maxpool,  # [B, 64, H/4, W/4]
        #     resnet.layer1,   # [B, 256, H/4, W/4]
        #     resnet.layer2,   # [B, 512, H/8, W/8]
        #     resnet.layer3,   # [B, 1024, H/16, W/16]
        #     resnet.layer4    # [B, 2048, H/16, W/16]（dilation = 2）
        # )
        self.backbone = BackBone(resnet)

        # 2. Segmentation Head
        self.aspp = ASPP(in_channels=2048, out_channels=256, rates=[1, 5, 10, 15])
        self.decoder = Decoder(low_level_dim=256, x_dim=256, num_classes=num_classes)

    def forward(self, x):
        input_size = x.shape[2:]  # 原图大小
        x, mid_feat = self.backbone(x)      # 特征图 输出为 [B, C, H/8, W/8] [B, 256, H/4, W/4]
        x = self.aspp(x)  # [B, 256, H/8, W/8]
        x = self.decoder(x, mid_feat)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x

if __name__ == '__main__':
    model = DeepLabV3p(num_classes=21)
    model.eval()
    # x = torch.randn(1, 3, 256, 256)
    # y = model(x)
    # print(y.shape)

    x = torch.randn(1, 3, 320, 320)
    inputs = x
    flops = FlopCountAnalysis(model, inputs)
    print(f"Total FLOPs: {(flops.total() / 1e9):.3f} G")

    # model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None)
    # model.eval()
    # x = torch.randn(1, 3, 320, 320)
    # inputs = x
    # flops = FlopCountAnalysis(model, inputs)
    # print(f"Total FLOPs: {(flops.total() / 1e9):.3f} G")
    # 打印每一层的 FLOPs（表格形式）
    print(flop_count_table(flops))