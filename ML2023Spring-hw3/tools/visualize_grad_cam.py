from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.abspath('..'))
from model.ResNet_a import *
from model.ResNet import *

# 用于保存特征和梯度
features = None
gradients = None

def forward_hook(module, input, output):
    global features
    features = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def grad_cam():
    # 全局平均池化梯度
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # shape: [1, C, 1, 1]
    # 加权求和得到 Grad-CAM
    grad_cam = torch.sum(weights * features, dim=1).squeeze()  # shape: [H, W]
    grad_cam = torch.relu(grad_cam)  # ReLU
    # 归一化
    grad_cam = grad_cam - grad_cam.min()
    grad_cam = grad_cam / grad_cam.max()
    grad_cam_np = grad_cam.detach().cpu().numpy()
    return grad_cam_np

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = resnet50(num_classes=11).to(device)
    model.load_state_dict(
        torch.load('../cpt/res50_82.19697.pt'),
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

    target_layer = model.layer2[1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)
    
    # 读取图片
    id_list = ['0_165', '3_5021', '5_9332', '7_1709', '8_2907', '9_253', '10_1696']
    for id in id_list:    
        img_pth = f'/root/autodl-tmp/MLCourseDataset/ml2023spring-hw3/train/{id}.jpg'
        raw_img = Image.open(img_pth).convert('RGB').resize((img_sz, img_sz))
        image = transform_test(raw_img).unsqueeze(0).to(device)
        
        # 前向传播
        output = model(image)
        pred_class = output.argmax().item()  # 预测类别
        # 反向传播（对预测类别求导）
        model.zero_grad()
        output[0, pred_class].backward()

        grad_cam_np = grad_cam()  # [H, W]
        # 将热力图调整为原图大小
        heatmap = cv2.resize(grad_cam_np, (img_sz, img_sz))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # PIL图像转OpenCV格式
        img_np = np.array(raw_img)  # RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # 叠加热力图
        alpha = 0.39  # 透明度
        overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)
        
        # 保存结果 (BGR格式)
        cv2.imwrite(f'visualization/GradCAM_{id}_{1}.png', overlay)
        print('已保存')

    
if __name__ == '__main__':
    main()