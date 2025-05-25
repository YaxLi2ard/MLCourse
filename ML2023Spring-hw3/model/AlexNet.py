import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义AlexNet网络结构
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        # 特征提取部分（卷积层）
        self.conv = nn.Sequential(
            # 第一层：输入图像通道为3，输出通道96，卷积核11x11，步长为4，填充为2
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=3, stride=2),  # 池化层，缩小尺寸
            
            # 第二层卷积：输出通道256，卷积核5x5，填充为2（保持尺寸）
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 第三层卷积：输出通道384
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            # 第四层卷积：输出通道384
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            # 第五层卷积：输出通道256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # 分类器部分（全连接层）
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),  # 展平后进入全连接层，原始AlexNet输入图像是227x227
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout防止过拟合

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes)  # 最后一层输出类别数
        )

    def forward(self, x):
        x = self.conv(x)  # 经过卷积层
        x = torch.flatten(x, start_dim=1)  # 展平为全连接层输入
        x = self.classifier(x)  # 经过全连接层
        return x


if __name__ == '__main__':
    model = AlexNet()
    x = torch.rand([1, 3, 256, 256])
    y = model(x)
    print(y.shape)