import torch
import torch.nn as nn

# 定义VGG16网络结构
class VGG16(nn.Module):
    def __init__(self, num_classes=1000):  # 默认分类数为1000，可自行修改
        super(VGG16, self).__init__()

        # 卷积部分
        self.features = nn.Sequential(
            # Block 1：两个3x3卷积 + 最大池化
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全连接分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),  # 输入尺寸通常为224x224，池化后为7x7，这里输入为256x256，池化后为8x8
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)  # 卷积部分
        x = torch.flatten(x, start_dim=1)  # 展平为全连接层输入
        x = self.classifier(x)  # 分类器部分
        return x


if __name__ == '__main__':
    model = VGG16()
    x = torch.rand([1, 3, 256, 256])
    y = model(x)
    print(y.shape)