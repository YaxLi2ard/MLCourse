from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.abspath('..'))
from model.ResNet_a import *

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = resnet50_a(num_classes=11).to(device)
    model.load_state_dict(
        torch.load('../cpt/res50+trans_83.14394.pt'),
        strict=False
    )
    model.eval()
    
    img_sz = 256
    normalize_mean = [0.558034, 0.45292863, 0.3462093]
    normalize_std = [0.22920622, 0.23979892, 0.23905471]
    transform_test = transforms.Compose([
        transforms.Resize(size=(img_sz, img_sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    
    # 读取图片
    id_list = ['0_71', '0_111', '0_165', '3_1777', '3_5021', '3_9910', '5_9779', '5_9950', '5_9332', '7_1709', '7_5637',
               '7_9130', '8_1069', '8_9007', '8_2907', '9_3399', '9_6116', '9_253', '10_1696', '10_1213', '10_601']
    for id in id_list:
        img_pth = f'/root/autodl-tmp/MLCourseDataset/ml2023spring-hw3/train/{id}.jpg'
        raw_img = Image.open(img_pth).convert('RGB').resize((img_sz, img_sz))
        image = transform_test(raw_img).unsqueeze(0).to(device)
        
        # 经过模型推理
        with torch.no_grad():
            output = model(image)
            
        attn = model.sa.last_attn_weights # [1, 65, 65]
        attn = attn[0, 0, 1:]
        attn_map = attn.reshape(16, 16).unsqueeze(0).unsqueeze(0)  # [1,1,16,16]
    
        # 使用双线性插值上采样到与原图相同尺寸
        attn_map = F.interpolate(attn_map, size=(img_sz, img_sz), mode='bilinear', align_corners=False)
        attn_map = attn_map.squeeze().cpu().numpy()  # [256,256]
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())  # 归一化
    
        # 转为热力图，0-255 uint8
        heatmap = np.uint8(255 * attn_map)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # BGR热力图
        alpha = 0.39  # 透明度
        # 叠加热力图到原图
        img_np = np.array(raw_img)  # RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
        # 保存结果 (BGR格式)
        cv2.imwrite(f'visualization/sa_{id}.png', overlay)
        print('已保存')

    
if __name__ == '__main__':
    main()