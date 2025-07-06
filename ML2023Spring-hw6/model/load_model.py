import torch
from model import networks
from model import legacy
import torchvision
import torch.nn as nn

pretrained_model_pth = './model/ffhq.pkl'

def load_partial(model, pretrained_model):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # print(f"共{len(pretrained_model.state_dict().items())}个参数层")
    # print(f"迁移了{len(pretrained_dict)}个参数层")
    return model

def load_generator(img_sz=64, pretrained=False):
    G = networks.Generator(
        z_dim=512, c_dim=0, w_dim=512,
        img_resolution=img_sz, img_channels=3
    )
    if pretrained:
        with open(pretrained_model_pth, 'rb') as f:
            pretrained_model = legacy.load_network_pkl(f)['G_ema']
        G = load_partial(G, pretrained_model)
    return G

def load_discriminator(img_sz=64, pretrained=False):
    D = networks.Discriminator(
        c_dim=0, img_resolution=img_sz, img_channels=3,
        architecture='resnet',  # 结构类型
        channel_base=32768,  # 通道数基数
        channel_max=512,  # 单层最大通道数
        num_fp16_res=3,
    )
    if pretrained:
        with open(pretrained_model_pth, 'rb') as f:
            pretrained_model = legacy.load_network_pkl(f)['D']
        D = load_partial(D, pretrained_model)
    return D

def load_discriminator_resnet(img_sz=64, pretrained=False):
    if pretrained:
        model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    else:
        model = torchvision.models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

if __name__ == '__main__':
    pretrained_model_pth = './model/ffhq.pkl'
    D = load_discriminator(pretrained=True)

