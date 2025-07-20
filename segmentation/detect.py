import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import random
from model.FCN import FCN
from model.UNet import UNet
from model.UNet_ResNet import UNet_ResNet
from model.UNet_ import UNet_
from model.DeepLabV3 import DeepLabV3
from PIL import Image
import numpy as np
import os

normalize_mean = (0.456, 0.443, 0.409)
normalize_std = (0.231, 0.227, 0.233)
img_sz = (320, 320)
# Pascal VOC colormap
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
              [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
              [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
              [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
              [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

def detect(model, img_pth, save_pth='output.png'):
    model.eval()
    # 加载并预处理图片
    img = Image.open(img_pth).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(img_sz),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]
    # 前向推理
    with torch.no_grad():
        output = model(input_tensor)  # [1, num_classes, H, W]
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()  # [H, W]
    # 转为彩色图
    color_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(VOC_COLORMAP):
        color_mask[pred == label] = color
    # 保存彩色 mask 图像
    color_img = Image.fromarray(color_mask)
    color_img.save(save_pth)
    print(f"Saved colored prediction to {save_pth}")

def load_weights(model, cpt_pth):
    model.load_state_dict(torch.load(cpt_pth), strict=False)
    return model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fcn = FCN(num_classes=21, backbone='resnet50', pretrained=False).to(device)
    unet = UNet_ResNet(num_classes=21, backbone='resnet50', pretrained=False).to(device)
    deeplab = DeepLabV3(num_classes=21, backbone='resnet50', pretrained=False).to(device)
    
    fcn = load_weights(fcn, 'cpt/fcn.pt')
    unet = load_weights(unet, 'cpt/unet_resnet50.pt')
    deeplab = load_weights(deeplab, 'cpt/deeplabv3.pt')

    img_root = '/root/autodl-tmp/MLCourseDataset/pascalvoc/VOCdevkit/VOC2012/JPEGImages'
    img_name = '2011_003011'
    save_root = 'output'
    img_pth = os.path.join(img_root, img_name + '.jpg')

    detect(fcn, img_pth, save_pth=os.path.join(save_root, f'{img_name}_fcn.png'))
    detect(unet, img_pth, save_pth=os.path.join(save_root, f'{img_name}_unet.png'))
    detect(deeplab, img_pth, save_pth=os.path.join(save_root, f'{img_name}_deeplab.png'))