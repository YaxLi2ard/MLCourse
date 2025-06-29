import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from tqdm import tqdm
from model.AlexNet import *
from model.ResNet import *
from model.ResNet_a import *
from model.VGGNet import *
from dataset import *
from utils import *

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_sz = 256
    normalize_mean = [0.558034, 0.45292863, 0.3462093]
    normalize_std = [0.22920622, 0.23979892, 0.23905471]
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=(img_sz, img_sz), scale=(0.9, 1.0)),
        # transforms.Resize(size=(256, 256)),
        transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=60),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(img_sz, img_sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    
    data_val = '/root/autodl-tmp/MLCourseDataset/ml2023spring-hw3/valid'
    data_test = '/root/autodl-tmp/MLCourseDataset/ml2023spring-hw3/test'
    tta = 0
    dataset_val = FoodClsDataset(data_dir=data_val, mode='val', tta=tta, transform=transform_test, transform2=transform_train)
    dataset_test = FoodClsDataset(data_dir=data_test, mode='test', tta=tta, transform=transform_test, transform2=transform_train)
    
    num_workers = 16
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    
    
    # 网络、优化器、计算器
    model = resnet50_a(num_classes=11).to(device)
    model.load_state_dict(
        torch.load('./cpt/res50+cbam_77.68939.pt'),
        strict=False
    )
    model.eval()

    dataloader = dataloader_test
    all_preds = []
    all_labels = []
    with torch.no_grad():
        pbar = tqdm(total=len(dataloader), desc="预测")
        for batch_idx, data in enumerate(dataloader):
            x, y = data
            x, y = x.to(device), y.to(device)  # x noTTA [1, 3, h, w] TTA [1, TTA, 3, h, w]
            if tta > 0:
                x = x.squeeze(dim=0)  # [TTA, 3, h, w]
                yp = model(x)  # [TTA, 11]
                yp_1 = yp[0]  # [11]
                yp_tta = yp[1:]  # [TTA-1, 11]
                yp_tta = torch.mean(yp_tta, dim=0)  # [11]
                alpha = 0.3
                yp = alpha * yp_1 + (1 - alpha) * yp_tta
                yp = yp.unsqueeze(dim=0)  # [1, 11]
                pred = torch.argmax(yp, dim=1)  # 预测的类别 [1]
            else:
                yp = model(x)  # [1, 11]
                pred = torch.argmax(yp, dim=1)  # 预测的类别 [1]
                
            all_preds.append(pred.cpu())   # 移到 CPU 并按顺序收集起来
            all_labels.append(y.cpu())
            
            pbar.update(1)
        pbar.close()
    
    # 拼接所有预测并转换为np数组
    all_preds = torch.cat(all_preds, dim=0)
    all_preds = all_preds.numpy()
    print('预测结果：', all_preds.shape)

    all_labels = torch.cat(all_labels, dim=0)
    all_labels = all_labels.numpy()
    correct = (all_preds == all_labels).sum()
    total = all_labels.shape[0]
    acc = correct / total
    print(f'准确率：{acc:.5f}', )
    
    # 填入csv文件
    output_csv(all_preds, output_path='submission.csv')


def output_csv(all_preds, output_path='submission.csv'):
    # 构造 Id 列：从 '0000' 开始依次递增
    ids = [f"{i:04d}" for i in range(len(all_preds))]
    # 构造 DataFrame
    df = pd.DataFrame({
        'Id': ids,
        'Category': all_preds
    })
    # 保存为 CSV 文件（不保存索引）
    df.to_csv(output_path, index=False)
    
if __name__ == '__main__':
    main()