import torch
import torchvision.models as tv_models
import timm
import re

def load_resnet(model):
    resnet_pretrained = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
    load_info = model.backbone.load_state_dict(resnet_pretrained.state_dict(), strict=False)
    # 打印未加载成功的参数名
    print("❌ 未加载成功的参数（模型需要但权重中没有）:")
    print(load_info.missing_keys)
    # 打印多余的参数（权重中有但模型中没有）
    print("⚠️ 多余的参数（权重中存在但模型中没有用到）:")
    print(load_info.unexpected_keys)
    print("Loaded weights successfully.")
    return model


def load_swin(model, type):
    # swin = timm.create_model(f'swin_{type}_patch4_window7_224', pretrained=True)
    swin = timm.create_model(f'swin_{type}_patch4_window7_224.ms_in22k_ft_in1k', pretrained=True)
    pretrained_state_dict = swin.state_dict()
    new_state_dict = {}
    # timm 的 swin 和 使用的 swin 中的 patch merging 索引编号错位了，因为timm里的patch merging 分给了当前 layer，而标准实现是分给了下一个layer
    for k, v in pretrained_state_dict.items():
        import re
        m = re.match(r'(layers)\.(\d+)\.(downsample\..+)', k)
        if m:
            prefix, idx_str, suffix = m.groups()
            idx = int(idx_str) - 1  # 索引+1
            new_key = f"{prefix}.{idx}.{suffix}"
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    load_info = model.backbone.load_state_dict(new_state_dict, strict=False)
    print("❌ 未加载成功的参数（模型需要但权重中没有）:")
    print(load_info.missing_keys)
    # 打印多余的参数（权重中有但模型中没有）
    print("⚠️ 多余的参数（权重中存在但模型中没有用到）:")
    print(load_info.unexpected_keys)
    print("Loaded weights successfully.")
    return model