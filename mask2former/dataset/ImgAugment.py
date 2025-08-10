import albumentations as A
from albumentations.core.transforms_interface import DualTransform
import numpy as np
import cv2
import math
import os
from PIL import Image
import numpy as np

class RandomScaleAndCrop:
    """
    随机缩放图像和mask（0.5~2倍），然后裁剪回原始尺寸（不足部分padding），裁剪位置随机
    img和mask共享同一组随机参数
    """

    def __init__(self, scale_limit=(0.5, 2.0)):
        self.scale_limit = scale_limit

    def get_params(self, image_shape):
        h, w = image_shape[:2]

        scale = np.random.uniform(*self.scale_limit)
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))

        pad_h = max(h - new_h, 0)
        pad_w = max(w - new_w, 0)

        top = np.random.randint(0, pad_h + 1) if pad_h > 0 else 0
        bottom = pad_h - top
        left = np.random.randint(0, pad_w + 1) if pad_w > 0 else 0
        right = pad_w - left

        final_h = new_h + pad_h
        final_w = new_w + pad_w

        start_y = np.random.randint(0, final_h - h + 1) if final_h - h > 0 else 0
        start_x = np.random.randint(0, final_w - w + 1) if final_w - w > 0 else 0

        return {
            "scale": scale,
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
            "start_y": start_y,
            "start_x": start_x,
            "orig_h": h,
            "orig_w": w,
            "new_h": new_h,
            "new_w": new_w
        }

    def apply(self, img, params, is_mask=False):
        interp = cv2.INTER_NEAREST if is_mask or img.ndim == 2 else cv2.INTER_LINEAR

        # 1. 缩放
        x = cv2.resize(img, (params["new_w"], params["new_h"]), interpolation=interp)

        # 2. padding（不居中）
        if params["top"] > 0 or params["bottom"] > 0 or params["left"] > 0 or params["right"] > 0:
            x = cv2.copyMakeBorder(
                x,
                params["top"], params["bottom"],
                params["left"], params["right"],
                cv2.BORDER_CONSTANT, value=0
            )

        # 3. 裁剪回原始大小
        x = x[params["start_y"]:params["start_y"] + params["orig_h"],
              params["start_x"]:params["start_x"] + params["orig_w"]]

        return x

    def __call__(self, image, mask):
        # 先生成参数
        params = self.get_params(image.shape)

        # 分别用同一参数变换
        image_aug = self.apply(image, params, is_mask=False)
        mask_aug = self.apply(mask, params, is_mask=True)

        return {'image': image_aug, 'mask': mask_aug}


class ResizeToFit(DualTransform):
    """
    等比例缩放图像和mask到尽量接近目标尺寸（不能超过），不足部分padding到指定大小
    """
    def __init__(self, target_h, target_w, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.target_h = target_h
        self.target_w = target_w

    def apply(self, img, **params):
        return self._resize_and_pad(img, interpolation=cv2.INTER_LINEAR)

    def apply_to_mask(self, mask, **params):
        return self._resize_and_pad(mask, interpolation=cv2.INTER_NEAREST)

    def _resize_and_pad(self, x, interpolation):
        h, w = x.shape[:2]
        scale = min(self.target_h / h, self.target_w / w, 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        x = cv2.resize(x, (new_w, new_h), interpolation=interpolation)

        pad_h = self.target_h - new_h
        pad_w = self.target_w - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        x = cv2.copyMakeBorder(x, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return x