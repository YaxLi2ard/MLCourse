import torch
from torch import nn, optim
import numpy as np
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import sys
import os
sys.path.append(os.path.abspath('..'))
from model.AlexNet import *
from model.ResNet import *
from model.ResNet_a import *
from model.VGGNet import *
from model.GoogleNet import *
from model.ResNeXt import *


if __name__ == '__main__':
    device = torch.device("cuda")
    # model = AlexNet(num_classes=11).to(device)
    model = VGG16(num_classes=11).to(device)
    # model = GoogleNet(num_classes=11, aux_logits=True).to(device)
    # model = resnext50(num_classes=11).to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    inputs = x

    model.eval()
    flops = FlopCountAnalysis(model, inputs)
    print(f"Total FLOPs: {(flops.total() / 1e9):.3f} G")