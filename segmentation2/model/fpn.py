import numpy as np
import torch
import torch.nn as nn

class FPNHead(nn.Module):
    def __init__(self, feature_strides=[4, 8, 16, 32], in_channels=[256, 512, 1024, 2048], channels=256,
                 kernel_size=3, align_corners=True):
        super(FPNHead, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))  # 每个尺度要上采样几次到最大尺度
            scale_head = []

            for k in range(head_length):  # k次 conv + upsample
                scale_head.append(
                    nn.Conv2d(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        kernel_size,
                        padding=kernel_size//2))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

    def forward(self, inputs):
        x = inputs[-len(inputs):]
        output = self.scale_heads[0](x[0])  # 先把最大尺寸经过连接层变为 output
        for i in range(1, len(self.feature_strides)):
            # self.scale_heads[i](x[i]) 后可能不是 output 尺寸，所以用插值精准调整到 output 尺寸
            output = output + nn.functional.interpolate(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        return output
        
        # outputs = []
        # x = inputs[-len(inputs):]
        # output = self.scale_heads[0](x[0])  # 先把最大尺寸经过连接层变为 output
        # for i in range(1, len(self.feature_strides)):
        #     # self.scale_heads[i](x[i]) 后可能不是 output 尺寸，所以用插值精准调整到 output 尺寸
        #     output = output + nn.functional.interpolate(
        #         self.scale_heads[i](x[i]),
        #         size=output.shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)

        # return output

class FPNHead_(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 2048], channels=256, kernel_size=3, align_corners=True):
        super(FPNHead_, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners
        # 横向 1x1 卷积（将不同层的通道统一到 channels）
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Conv2d(in_ch, channels, kernel_size=1)
            )
        # 输出 3x3 卷积
        self.output_convs = nn.ModuleList()
        for _ in in_channels:
            self.output_convs.append(
                nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
            )

    def forward(self, inputs):
        # 1. 横向卷积，统一通道数
        feats = [l_conv(x) for x, l_conv in zip(inputs, self.lateral_convs)]
        # 2. 自顶向下融合（从低分辨率往高分辨率逐步上采样）
        for i in range(len(feats) - 1, 0, -1):  
            # 将上一层(低分辨率)上采样到当前层的尺寸
            upsample = nn.functional.interpolate(feats[i], size=feats[i - 1].shape[2:], mode='bilinear', align_corners=self.align_corners)
            feats[i - 1] = feats[i - 1] + upsample  # 融合

        # 3. 每个尺度都用 3x3 卷积平滑
        outputs = [o_conv(f) for f, o_conv in zip(feats, self.output_convs)]

        return outputs

if __name__ == '__main__':
    x = [torch.randn(1, 256, 160, 160), torch.randn(1, 512, 80, 80), torch.randn(1, 1024, 40, 40), torch.randn(1, 2048, 20, 20)]
    fpn = FPNHead()
    y = fpn(x)
    print(y.shape)