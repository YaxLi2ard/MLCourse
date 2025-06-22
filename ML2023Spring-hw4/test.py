import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from model.Conformer import Conformer
from model.Conformer2 import Conformer2
from model.Transformer import Transformer
from model.SwinTransformer import SwinTransformer
from DatasetManager import *
from utils import *

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = '/root/autodl-tmp/MLCourseDataset/ml2023spring-hw4/Dataset'
    dataset_manager = DatasetManager(data_dir, valid_ratio=0.1, segment_len=128)
    dataset_train, dataset_val, dataset_val2, dataset_test, dataset_test2 = dataset_manager.get_datasets()
    id2speaker = dataset_manager.get_id2speaker()
    file_ids = dataset_manager.test_files
    dataset = dataset_test2
    # 网络
    # model = Conformer().to(device)
    # model = SwinTransformer().to(device)
    model = Transformer().to(device)
    model.load_state_dict(
        torch.load('./cpt/trans_89.73002.pt'),
        strict=False
    )
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        pbar = tqdm(total=len(dataset), desc="预测")
        for batch_idx, data in enumerate(dataset):
            x, y = data
            x, y = x.to(device), y.to(device)  # x [t, 40]
            x = x.unsqueeze(dim=0)  # x [1, t, 40]
            
            yp = model(x)  # [1, 600]
            pred = torch.argmax(yp, dim=1)  # 预测的类别 [1]
            
            all_preds.append(pred.cpu())   # 移到 CPU 并按顺序收集起来
            all_labels.append(y.cpu())
            
            pbar.update(1)
        pbar.close()
    
    # 拼接所有预测并转换为np数组
    all_preds = torch.cat(all_preds, dim=0)
    all_preds = all_preds.numpy()
    print('预测结果：', all_preds.shape)

    all_labels = np.array(all_labels)
    correct = (all_preds == all_labels).sum()
    total = all_labels.shape[0]
    acc = correct / total
    print(f'准确率：{acc:.5f}', )
    
    # 填入csv文件
    all_preds = [id2speaker[id] for id in all_preds]
    output_csv(file_ids, all_preds, output_path='submission.csv')


def output_csv(file_ids, all_preds, output_path='submission.csv'):
    # 构造 DataFrame
    df = pd.DataFrame({
        'Id': file_ids,
        'Category': all_preds
    })
    # 保存为 CSV 文件（不保存索引）
    df.to_csv(output_path, index=False)
    
if __name__ == '__main__':
    main()