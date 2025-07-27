import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, parameter_count_table

# BasicBlock
class BasicBlock(nn.Module):
    expansion = 1  # 不改变通道数

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# 模块 一个stage，多分辨率分支分别经过block后融合
class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, block, num_blocks, num_channels):
        '''
        :param num_branches: 分辨率分支数量
        :param block: 特征提取的block
        :param num_blocks: 列表，每个分支的block数量
        :param num_channels: 列表，每个分支进入的通道数，也是输出的通道数
        '''
        super().__init__()
        self.num_branches = num_branches
        # 每个分支的特征提取模块
        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        # 融合的操作，两两分支之间用上采样或下采样或无操作（同一分辨率分支）
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels):
        layers = []
        for i in range(num_blocks[branch_index]):
            layers.append(block(num_channels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        return nn.ModuleList([
            self._make_one_branch(i, block, num_blocks, num_channels)
            for i in range(num_branches)
        ])

    def _make_fuse_layers(self):
        # 构建融合层（把所有分支的输出调整为同一尺度再相加）
        fuse_layers = []
        for i in range(self.num_branches):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:  # i=i+j j的分辨率小于i因此对j上采样
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(
                            in_channels=self.branches[j][-1].bn2.num_features,
                            out_channels=self.branches[i][-1].bn2.num_features,
                            kernel_size=1, bias=False),
                        nn.BatchNorm2d(self.branches[i][-1].bn2.num_features, momentum=0.1),
                        nn.Upsample(scale_factor=2**(j - i), mode='nearest')
                    ))
                elif j == i:  # 不操作
                    fuse_layer.append(None)
                else:  # i=i+j j的分辨率大于i因此对j下采样
                    # 下采样
                    down_layers = []
                    for k in range(i - j - 1):
                        down_layers.append(nn.Sequential(
                            nn.Conv2d(
                                in_channels=self.branches[j][-1].bn2.num_features,
                                out_channels=self.branches[j][-1].bn2.num_features,
                                kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(self.branches[j][-1].bn2.num_features, momentum=0.1),
                            nn.ReLU(inplace=True)
                        ))
                    down_layers.append(nn.Sequential(
                        nn.Conv2d(
                            in_channels=self.branches[j][-1].bn2.num_features,
                            out_channels=self.branches[i][-1].bn2.num_features,
                            kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(self.branches[i][-1].bn2.num_features, momentum=0.1),
                    ))
                    fuse_layer.append(nn.Sequential(*down_layers))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        # x: 不同分辨率特征图的列表
        # 每个分支的特征提取
        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        # 融合
        x_fused = []
        for i in range(self.num_branches):
            y = x[i]
            for j in range(self.num_branches):
                if i == j:
                    continue
                if self.fuse_layers[i][j] is not None:
                    if j > i:
                        y = y + self.fuse_layers[i][j](x[j])
                    else:
                        y = y + self.fuse_layers[i][j](x[j])
            x_fused.append(self.relu(y))
        return x_fused


def HRNet(num_classes=21, use_OCR=False, pretrained=True):
    model = HighResolutionNet(num_classes=num_classes, use_OCR=use_OCR)
    if pretrained:
        weights = torch.load('./model/hrnet_w32-36af842e.pth')
        load_info = model.load_state_dict(weights, strict=False)
    return model
    

# HRNet v2
class HighResolutionNet(nn.Module):
    def __init__(self, num_classes=21, use_OCR=False):
        super().__init__()
        # Stem 先下采样4倍
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        # Stage1 [1/4, 1/4, 256]
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.1)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )
        # Stage2 - [1/4, 1/4, 32] [1/8, 1/8, 64]
        self.transition1 = self._make_transition_layer([256], [32, 64])
        self.stage2 = nn.Sequential(
            HighResolutionModule(num_branches=2, block=BasicBlock, num_blocks=[4, 4], num_channels=[32, 64])
        )
        # Stage3 - [1/4, 1/4, 32] [1/8, 1/8, 64] [1/16, 1/16, 128]
        self.transition2 = self._make_transition_layer([32, 64], [32, 64, 128])
        self.stage3 = nn.Sequential(
            HighResolutionModule(num_branches=3, block=BasicBlock, num_blocks=[4, 4, 4], num_channels=[32, 64, 128]),
            HighResolutionModule(num_branches=3, block=BasicBlock, num_blocks=[4, 4, 4], num_channels=[32, 64, 128]),
            HighResolutionModule(num_branches=3, block=BasicBlock, num_blocks=[4, 4, 4], num_channels=[32, 64, 128]),
            HighResolutionModule(num_branches=3, block=BasicBlock, num_blocks=[4, 4, 4], num_channels=[32, 64, 128]),
        )
        # Stage4 - [1/4, 1/4, 32] [1/8, 1/8, 64] [1/16, 1/16, 128] [1/32, 1/32, 256]
        self.transition3 = self._make_transition_layer([32, 64, 128], [32, 64, 128, 256])
        self.stage4 = nn.Sequential(
            HighResolutionModule(num_branches=4, block=BasicBlock, num_blocks=[4, 4, 4, 4], num_channels=[32, 64, 128, 256]),
            HighResolutionModule(num_branches=4, block=BasicBlock, num_blocks=[4, 4, 4, 4], num_channels=[32, 64, 128, 256]),
            HighResolutionModule(num_branches=4, block=BasicBlock, num_blocks=[4, 4, 4, 4], num_channels=[32, 64, 128, 256]),
        )
        # Final head
        self.use_OCR = use_OCR
        f_dim = 32 + 64 + 128 + 256
        if use_OCR:
            self.ocr = OCRModule(in_channels=f_dim, mid_channels=512, num_classes=num_classes)
        else:
            self.head = nn.Sequential(
                nn.Conv2d(f_dim, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_classes, kernel_size=1)
            )

    def _make_transition_layer(self, prev_channels, curr_channels):
        '''
        :param prev_channels: 列表，transition前每个分支的通道数
        :param curr_channels:  列表，transition后每个分支的通道数
        '''
        layers = []
        for i in range(len(curr_channels)):
            if i < len(prev_channels):  # 新分支对应旧分支
                if prev_channels[i] != curr_channels[i]:  # 通道数不一致则卷积改变通道数
                    layers.append(nn.Sequential(
                        nn.Conv2d(prev_channels[i], curr_channels[i], 3, padding=1, bias=False),
                        nn.BatchNorm2d(curr_channels[i], momentum=0.1),
                        nn.ReLU(inplace=True)
                    ))
                else:  # 新分支与旧分支通道数一致则不操作
                    layers.append(None)
            else:  # 下采样前一层最后一个产生新分辨率分支
                # 新分支，下采样前一层最后一个
                downsample = nn.Sequential(
                    nn.Conv2d(prev_channels[-1], curr_channels[i], 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(curr_channels[i], momentum=0.1),
                    nn.ReLU(inplace=True)
                )
                layers.append(nn.Sequential(downsample))
        return nn.ModuleList(layers)

    def forward(self, x):
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        # Stage1
        x = self.layer1(x)
        # Stage2
        x_list = [trans(x) if trans else x for trans in self.transition1]
        x_list = self.stage2(x_list)
        # Stage3
        x_list = [trans(xi) if trans else xi for xi, trans in zip(x_list + [x_list[-1]], self.transition2)]
        x_list = self.stage3(x_list)
        # Stage4
        x_list = [trans(xi) if trans else xi for xi, trans in zip(x_list + [x_list[-1]], self.transition3)]
        x_list = self.stage4(x_list)
        # Upsample & concat
        for i in range(1, len(x_list)):
            x_list[i] = F.interpolate(x_list[i], size=x_list[0].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat(x_list, dim=1)

        if self.use_OCR:
            out1, out2 = self.ocr(x)
            out1 = F.interpolate(out1, scale_factor=4, mode='bilinear', align_corners=False)  # 恢复到原图大小
            out2 = F.interpolate(out2, scale_factor=4, mode='bilinear', align_corners=False)
            out = [out1, out2]
        else:
            out = self.head(x)
            out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)  # 恢复到原图大小
        return out

# OCR模块
class SpatialGatherModule(nn.Module):
    """
    将每个像素的信息聚合为每个类别的上下文特征 类似ROI Pooling
    """
    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale

    def forward(self, feats, probs):
        # feats: [N, C, H, W]；probs: [N, K, H, W]
        N, C, H, W = feats.shape
        probs = probs.view(N, -1, H * W)  # [N, K, H*W]
        feats = feats.view(N, C, H * W)   # [N, C, H*W]

        probs = F.softmax(self.scale * probs, dim=2)
        context = torch.bmm(feats, probs.permute(0, 2, 1))  # [N, C, K]
        context = context.permute(0, 2, 1)     # [N, K, C]
        return context

class ObjectAttentionBlock(nn.Module):
    """
    使用注意力机制将类别上下文信息融合回每个像素
    """
    def __init__(self, in_channels, key_channels, scale=1):
        super().__init__()
        self.scale = scale
        self.conv_query = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.conv_key = nn.Linear(in_channels, key_channels, bias=False)
        self.conv_value = nn.Linear(in_channels, in_channels, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, context):
        # x: [N, C, H, W]；context: [N, K, C]
        N, C, H, W = x.shape
        query = self.conv_query(x).view(N, -1, H * W).permute(0, 2, 1)  # [N, HW, C']
        key = self.conv_key(context).permute(0, 2, 1)                   # [N, C', K]
        value = self.conv_value(context).permute(0, 2, 1)               # [N, C, K]

        sim_map = torch.bmm(query, key)  # [N, HW, K]
        sim_map = self.softmax(sim_map)

        context = torch.bmm(value, sim_map.permute(0, 2, 1))  # [N, C, HW]
        context = context.view(N, C, H, W)

        out = torch.cat([x, context], dim=1)  # [N, 2C, H, W]
        return out

class OCRModule(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.aux_head = nn.Sequential(  # 辅助头，用于指导 OCR 上下文
            nn.Conv2d(in_channels, 128, 1, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        self.gather = SpatialGatherModule()
        self.object_attention = ObjectAttentionBlock(mid_channels, key_channels=mid_channels // 2)
        self.last_conv = nn.Sequential(
            nn.Conv2d(2 * mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1)
        )

    def forward(self, feats):
        preds = self.aux_head(feats)
        feats = self.conv3x3(feats)                 # 转换通道数
        context = self.gather(feats, preds)         # [N, K, C]
        feats = self.object_attention(feats, context)
        out = self.last_conv(feats)
        return out, preds

if __name__ == '__main__':
    model = HRNet(num_classes=21, use_OCR=True, pretrained=False)
    # x = torch.randn(1, 3, 320, 320)
    # print(model(x).shape)

    inputs = torch.randn(1, 3, 320, 320)
    model.eval()
    flops = FlopCountAnalysis(model, inputs)
    print(f"Total FLOPs: {(flops.total() / 1e9):.3f} G")

    # weights = torch.load('hrnet_w32-36af842e.pth')
    # load_info = model.load_state_dict(weights, strict=False)
    # # 打印未加载成功的参数名
    # print("❌ 未加载成功的参数（模型需要但权重中没有）:")
    # print(load_info.missing_keys)
    # # 打印多余的参数（权重中有但模型中没有）
    # print("⚠️ 多余的参数（权重中存在但模型中没有用到）:")
    # print(load_info.unexpected_keys)
    # print("Loaded weights successfully.")
    # print(weights.keys())