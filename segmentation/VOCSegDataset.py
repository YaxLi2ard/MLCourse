import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np

class VOCSegDataset(Dataset):
    def __init__(self, root, image_set='train', transform=None):
        """
        :param root: VOCdevkit 根目录，如 'VOCdevkit/VOC2012'
        :param image_set: 'train', 'val' 或 'trainval'
        :param transform: 图像变换操作（同时应用于图像和掩码）
        """
        self.root = root
        self.image_set = image_set
        self.transform = transform

        # 读取 train.txt 或 val.txt 中的图像 ID
        splits_file = os.path.join(root, 'ImageSets/Segmentation', image_set + '.txt')
        with open(splits_file, 'r') as f:
            self.image_ids = [x.strip() for x in f.readlines()]

        self.image_dir = os.path.join(root, 'JPEGImages')
        self.mask_dir = os.path.join(root, 'SegmentationClass')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        mask_path = os.path.join(self.mask_dir, image_id + '.png')

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)  # 不转RGB，保持每像素为类索引

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask

    @staticmethod
    def collate_fn(batch):
        """
        对不定大小图像进行padding，图像用0填充，mask用255（表示ignore index）填充
        """
        images, masks = list(zip(*batch))
        batched_images = cat_list(images, fill_value=0)
        batched_masks = cat_list(masks, fill_value=255)
        return batched_images, batched_masks


def cat_list(images, fill_value=0):
    """
    将任意大小的图像/掩码拼接成batch，并用指定数值填充空白部分
    :param images: list of tensors [C, H, W] 或 [1, H, W]
    :param fill_value: 图像一般为0，标签为255
    :return: batched_tensor: [B, C, H_max, W_max]
    """
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new_full(batch_shape, fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


if __name__ == '__main__':
    pass