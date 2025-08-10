import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import DualTransform
import numpy as np
import cv2
import math
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from .ImgAugment import *
from collections import namedtuple


# Cityscapes 训练用类别映射表（原始ID -> 训练ID）
# 参考：https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


class CityscapesDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        """
        Cityscapes 数据集读取类
        参数:
        - root (str): Cityscapes 数据集根目录
        - mode (str): 数据划分类型，可为 'train', 'val', 'test'
        - transform (callable, optional): 对图像和标签的联合变换（图像增强）
        """
        self.root = root
        self.mode = mode
        self.transform = transform

        # 图像和标签的路径
        self.image_dir = os.path.join(root, 'leftImg8bit', mode)
        self.label_dir = os.path.join(root, 'gtFine', mode)

        self.image_paths = []
        self.label_paths = []

        # 遍历所有城市
        for city in os.listdir(self.image_dir):
            img_dir = os.path.join(self.image_dir, city)
            lbl_dir = os.path.join(self.label_dir, city)
            for file_name in os.listdir(img_dir):
                if file_name.endswith('_leftImg8bit.png'):
                    # 构造图像和标签路径
                    img_path = os.path.join(img_dir, file_name)
                    label_file = file_name.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                    lbl_path = os.path.join(lbl_dir, label_file)

                    self.image_paths.append(img_path)
                    self.label_paths.append(lbl_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        加载第 index 个样本（图像和标签）

        返回:
        - image (Tensor or PIL.Image): 原始图像
        - label (Tensor or PIL.Image): 对应的语义标签图（像素值为类别 ID）
        """
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # 灰度图，每个像素是类别 ID
        # 转为numpy数组
        image = np.array(image)
        label = np.array(label)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']  # shape: [3, H, W]
            label = augmented['mask']  # shape: [H, W]
        return image, label.long()



if __name__ == '__main__':
    cityscapes_root = '/root/autodl-tmp/MLCourseDataset/cityscapes'
    img_size = [256, 512]
    transform = A.Compose([
        RandomScaleAndCrop(scale_limit=(0.5, 2.0)),
        ResizeToFit(target_h=img_size[0], target_w=img_size[1], p=1.0),
        A.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.2, 0.2), p=0.8),
        A.HorizontalFlip(p=0.5),
        ToTensorV2()
    ])
    val_dataset = CityscapesDataset(cityscapes_root, mode='train', transform=transform)

    print("数据集长度:", len(val_dataset))

    x, y = val_dataset[0]
    print("输入图像形状:", x.shape)
    print("标签图像形状:", y.shape)
    print(y.min(), y.max())
    