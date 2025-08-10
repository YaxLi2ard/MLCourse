import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.common.config import CfgNode
from configs.config import Config
import argparse
from modeling.MaskFormerModel import MaskFormerModel
from dataset.CityscapesDataset import CityscapesDataset
from dataset.CityscapesDataset import labels as cityspaces_mapper
from utils.draw import get_cityscapes_colormap, get_image_mask_overlay
from utils.load_pretrained import load_resnet, load_swin
import os
import cv2
from PIL import Image
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/maskformer_nuimages.yaml')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--project_name", default='NuImages_swin_base_Seg', type=str)

    args = parser.parse_args()
    cfg_ake150 = Config.fromfile(args.config)

    cfg_base = CfgNode.load_yaml_with_base(args.config, allow_unsafe=True)    
    cfg_base.update(cfg_ake150.__dict__.items())

    cfg = cfg_base
    for k, v in args.__dict__.items():
        cfg[k] = v

    cfg = Config(cfg)

    cfg.ngpus = torch.cuda.device_count()
    if torch.cuda.device_count() > 1:
        cfg.local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(cfg.local_rank)
    return cfg
    
normalize_mean = [0.28689553, 0.32513301, 0.28389176]
normalize_std = [0.18696375, 0.19017339, 0.18720214]

def detect(img_name):
    # 1. 构造文件路径
    img_path = os.path.join(data_root, "leftImg8bit", "val", img_name.split('_')[0], img_name + ".png")
    label_path = os.path.join(data_root, "gtFine", "val", img_name.split('_')[0], img_name.replace("_leftImg8bit", "_gtFine_labelTrainIds.png"))
    # 2. 读取原图和标签
    img = Image.open(img_path).convert('RGB')
    label = Image.open(label_path).convert('L')  # 灰度图，每个像素是类别 ID
    # 转为numpy数组
    img = np.array(img)
    label = np.array(label)
    # 3. resize
    img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
    # 4. 标准化 & 转 tensor
    img = img.astype(np.float32) / 255.0
    img = (img - normalize_mean) / normalize_std
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)  # [1, 3, H, W]
    label_tensor = torch.from_numpy(label).unsqueeze(0).to(device)  # [1, H, W]
    # 5. 模型前向推理
    with torch.no_grad():
        output = model(img_tensor)
    # 6. 后处理
    mask_img = post_process(output, filter=False, threshold=0.05)  # [b, h, w]
    mask_img[label_tensor == 255] = 255
    # 7. 叠加原图像
    overlay = get_image_mask_overlay(img_tensor[0], mask_img[0], normalize_mean, normalize_std, cityspaces_colormap, alpha=0.5)
    overlay = (overlay * 255).astype(np.uint8)
    overlay0 = get_image_mask_overlay(img_tensor[0], label_tensor[0], normalize_mean, normalize_std, cityspaces_colormap, alpha=0.5)
    overlay0 = (overlay0 * 255).astype(np.uint8)
    # 8. 保存结果
    os.makedirs(save_root, exist_ok=True)
    cv2.imwrite(os.path.join(save_root, f"{img_name}_pred.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_root, f"{img_name}_gt.png"), cv2.cvtColor(overlay0, cv2.COLOR_RGB2BGR))

def post_process(output, filter, threshold=0.5):
    mask_cls_results = output["pred_logits"]
    mask_pred_results = output["pred_masks"]
    # 上采样
    mask_pred_results = F.interpolate(
        mask_pred_results,
        scale_factor=cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
        mode="bilinear",
        align_corners=False,
    )
    pred_masks = semantic_inference(mask_cls_results, mask_pred_results)  # [b, num_cls, h, w]
    if filter:
        probs = torch.softmax(pred_masks, dim=1)  # [B, num_cls, H, W]
        # probs = pred_masks
        conf, _ = torch.max(probs, dim=1)  # [B, H, W]
        # print(conf[0,0:10,0:10])
        ignore_mask = conf < threshold  # bool [B, H, W]
        mask_img = torch.argmax(pred_masks, dim=1)  # [B, H, W]
        mask_img[ignore_mask] = 255  # 置信度低的区域变为255
    else:
        mask_img = torch.argmax(pred_masks, dim=1)  # [B, H, W]
    return mask_img
    

def semantic_inference(mask_cls, mask_pred):    
    # mask_cls [b, num_q, num_classed+1] mask_pred [b, num_q, h, w]
    mask_cls = F.softmax(mask_cls, dim=-1)[...,:-1]  # 去掉no-object类
    mask_pred = mask_pred.sigmoid()      
    semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)        
    return semseg

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = get_args()
    model = MaskFormerModel(cfg)
    model.to(device)
    model.load_state_dict(torch.load('./cpt/cityscapes_swin_base_100.pt'), strict=False)
    model.eval()

    # 掩码颜色映射
    cityspaces_colormap = get_cityscapes_colormap(labels=cityspaces_mapper, num_classes=19)
    
    img_size = [512, 1024]
    save_root = './output'
    data_root = '/root/autodl-tmp/MLCourseDataset/cityscapes'
    imgs_name = ['frankfurt_000000_001016_leftImg8bit', 'frankfurt_000001_070099_leftImg8bit', 'lindau_000003_000019_leftImg8bit',
                 'lindau_000033_000019_leftImg8bit', 'lindau_000056_000019_leftImg8bit', 'lindau_000011_000019_leftImg8bit',
                 'lindau_000019_000019_leftImg8bit', 'munster_000009_000019_leftImg8bit', 'munster_000061_000019_leftImg8bit',
                 'munster_000093_000019_leftImg8bit', 'munster_000161_000019_leftImg8bit', 'munster_000033_000019_leftImg8bit',
                 'frankfurt_000001_077233_leftImg8bit', 'frankfurt_000000_003357_leftImg8bit', 'frankfurt_000001_070099_leftImg8bit']
    for img_name in imgs_name:
        detect(img_name)