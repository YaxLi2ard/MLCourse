# from networks import *
#
# G = Generator(
#     z_dim=512, c_dim=0, w_dim=512,
#     img_resolution=64, img_channels=3
# )
# z = torch.randn([1, 512])
# img = G(z, None)
# print(img.shape)
#
# D = Discriminator(
#     c_dim=0, img_resolution=64, img_channels=3,
#     architecture='resnet',  # 结构类型
#     channel_base=32768,  # 通道数基数
#     channel_max=512,  # 单层最大通道数
#     num_fp16_res=0,
# )
# y = D(img, None)
# print(y.shape)

import torch
import networks
import legacy

with open('ffhq.pkl', 'rb') as f:
    pretrained = legacy.load_network_pkl(f)['G_ema']

G = networks.Generator(
    z_dim=512, c_dim=0, w_dim=512,
    img_resolution=64, img_channels=3
)

def load_partial(model, pretrained):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained.state_dict().items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"共{len(pretrained.state_dict().items())}个参数层")
    print(f"迁移了{len(pretrained_dict)}个参数层")

load_partial(G, pretrained)


z = torch.randn([1, 512])
img = G(z, None)
print(img.shape)






