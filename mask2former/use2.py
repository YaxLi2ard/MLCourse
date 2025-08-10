import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.common.config import CfgNode
from configs.config import Config
import argparse
from modeling.MaskFormerModel import MaskFormerModel
from dataset.VOCSegDataset import VOC_COLORMAP
from utils.draw import get_cityscapes_colormap, get_image_mask_overlay
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
    
normalize_mean = (0.456, 0.443, 0.409)
normalize_std = (0.231, 0.227, 0.233)

def detect(img_name):
    # 1. 构造文件路径
    img_path = os.path.join(img_root, img_name + ".jpg")
    # 2. 读取原图
    img = Image.open(img_path).convert('RGB')
    # 转为numpy数组
    img = np.array(img)
    # 3. resize
    img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
    # 4. 标准化 & 转 tensor
    img = img.astype(np.float32) / 255.0
    img = (img - normalize_mean) / normalize_std
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)  # [1, 3, H, W]
    # 5. 模型前向推理
    with torch.no_grad():
        output = model(img_tensor)
    # 6. 后处理
    mask_img = post_process(output, filter=False, threshold=0.05)  # [b, h, w]
    # 7. 叠加原图像
    overlay = get_image_mask_overlay(img_tensor[0], mask_img[0], normalize_mean, normalize_std, VOC_COLORMAP, alpha=1)
    overlay = (overlay * 255).astype(np.uint8)
    # 8. 保存结果
    os.makedirs(save_root, exist_ok=True)
    cv2.imwrite(os.path.join(save_root, f"{img_name}_{img_size[0]}_swin_pred.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

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
    model.load_state_dict(torch.load('./cpt/voc_swin_base_100.pt'), strict=False)
    model.eval()
    
    img_size = [320, 320]
    save_root = './output'
    img_root = '/root/autodl-tmp/MLCourseDataset/pascalvoc/VOCdevkit/VOC2012/JPEGImages'
    imgs_name = ['2007_000033', '2007_000063', '2007_000392', '2007_000837', '2007_001311', '2007_001955', '2007_002361', '2007_003131', '2007_005331', '2007_009096', 
                '2007_009923', '2008_000765', '2008_001715', '2008_003105', '2008_007507', '2009_000989', '2009_003003', '2009_003551', '2009_005302', '2010_000065', 
                '2010_000530', '2010_002030', '2010_003771', '2011_000226', '2011_001910', '2011_003011']
    for img_name in imgs_name:
        detect(img_name)