import torch
from torch import nn, optim
import numpy as np
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import sys
import os
from model.Conformer import *
from model.Transformer import *
from model.SwinTransformer import *



if __name__ == '__main__':
    device = torch.device("cuda")
    # model = Conformer().to(device)
    # model = Transformer().to(device)
    model = SwinTransformer().to(device)
    x = torch.randn(1, 128, 40).to(device)
    inputs = x

    model.eval()
    flops = FlopCountAnalysis(model, inputs)
    print(f"Total FLOPs: {(flops.total() / 1e9):.3f} G")

    # 参数量统计
    print("Parameter Count:")
    print(parameter_count_table(model))