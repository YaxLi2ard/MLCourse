from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table
from model.seg_model import SegModel
import torch

if __name__ == '__main__':
    model = SegModel(backbone='p2t', head='maskformer', pretrained=False).eval()
    x = torch.randn(1, 3, 320, 320)
    inputs = x
    flops = FlopCountAnalysis(model, inputs)
    print(f"Total FLOPs: {(flops.total() / 1e9):.3f} G")
    print(flop_count_table(flops))