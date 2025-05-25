import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def compute_mean_std(img_dir, resize=None):
    """
    统计图像文件夹中的所有图像的 mean 和 std
    """
    means = []
    stds = []

    img_list = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for fname in tqdm(img_list, desc="Processing images"):
        path = os.path.join(img_dir, fname)
        img = Image.open(path).convert('RGB')  # 转为 RGB 格式
        if resize:
            img = img.resize(resize)
        img = np.array(img).astype(np.float32) / 255.0  # [H, W, C] -> [0,1]

        means.append(np.mean(img, axis=(0, 1)))  # 每张图的三个通道均值
        stds.append(np.std(img, axis=(0, 1)))    # 每张图的三个通道标准差

    # 所有图像的通道均值和标准差的平均
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    print(f"Mean (R, G, B): {mean}")
    print(f"Std  (R, G, B): {std}")

    return mean, std


if __name__ == '__main__':
    img_dir = '/home/aistudio/data/data339209/train'
    compute_mean_std(img_dir, resize=(256, 256))