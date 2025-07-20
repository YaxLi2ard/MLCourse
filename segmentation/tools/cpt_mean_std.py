import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def compute_mean_std(root, resize=None):
    """
    统计图像文件夹中的所有图像的 mean 和 std
    """
    means = []
    stds = []

    # 读取 train.txt 中的图像 ID
    splits_file = os.path.join(root, 'ImageSets/Segmentation/train.txt')
    with open(splits_file, 'r') as f:
        image_ids = [x.strip() for x in f.readlines()]

    image_dir = os.path.join(root, 'JPEGImages')

    for fname in tqdm(image_ids, desc="Processing images"):
        path = os.path.join(image_dir, fname + '.jpg')
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
    root = '/root/autodl-tmp/MLCourseDataset/pascalvoc/VOCdevkit/VOC2012'
    compute_mean_std(root, resize=(256, 256))