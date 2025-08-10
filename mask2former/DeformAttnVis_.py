"""
Deformable Attention 可视化工具（基于 PyQt5）

功能：
- 浏览/选择图像（从 data_root）
- 调用已有模型进行 detect（前向推理），显示分割掩码（右侧）
- 选择 layer_id 与 lvl_id
- 在原图上点击某个像素点，计算该点对应特征图上的位置，读取可变形注意力的 sampling offsets
- 将各个头的采样点在原图上绘制出来（不同头不同颜色；不同 lvl 用大小区分）
- 支持保存带标注的图像

注意：
- 本脚本依赖你已有的模型定义与权重加载流程（和你在问题中给出的代码片段兼容）
- 假设 get_deform_offsets 返回的是张量形状为
  [B, H, W, n_head, n_lvl, n_points, 2] （以 feature map 像素为单位或为相对偏移，本脚本尝试智能识别并做缩放）
- 你需要安装 PyQt5: pip install PyQt5

作者：基于你提供的代码改写并加入 GUI（中文注释）
"""

import os
import sys
import math
import numpy as np
import cv2
from PIL import Image
from fvcore.common.config import CfgNode
from configs.config import Config
import argparse

import torch
import torch.nn.functional as F

# PyQt5 UI
from PyQt5 import QtCore, QtGui, QtWidgets

# --------------------------
# 这里把你原来那份脚本中的模型加载、preprocess、postprocess 的内容整合进来
# 需要确保 MaskFormerModel, Config 等定义在同一目录或可导入
# --------------------------
try:
    from configs.config import Config
    from modeling.MaskFormerModel import MaskFormerModel
except Exception as e:
    # 如果导入失败，提醒用户调整 PYTHONPATH 或把该脚本放在工程根目录
    print("导入模型定义失败：", e)
    print("请确保本文件在工程根目录下或 PYTHONPATH 包含工程路径，以便导入 configs/config.py 与 modeling/MaskFormerModel.py")


# --------------------------
# 你可以修改下面默认设置：
# --------------------------
IMG_SIZE = [320, 320]  # 你用于推理时的输入大小 (H, W)
NORMALIZE_MEAN = (0.456, 0.443, 0.409)
NORMALIZE_STD = (0.231, 0.227, 0.233)
FEATUREMAP_SZ = [[10, 10], [20, 20], [40, 40]]  # 与你代码一致
DATA_ROOT = 'Y:/Dataset/PASCALVOC/pascalvoc/VOCdevkit/VOC2012/JPEGImages'  # 修改为你的路径
WEIGHTS_PATH = './cpt/voc_swin_base_100.pt'

# 颜色表（用于不同 head 的绘制）
DEFAULT_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
]


# --------------------------
# 模型封装：加载模型并提供 detect() 与 get_deform_offsets_point() 接口
# --------------------------
class ModelWrapper:
    def __init__(self, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.model = MaskFormerModel(cfg)
        self.model.to(self.device)
        # 尝试加载权重（非严格模式）
        try:
            self.model.load_state_dict(torch.load(WEIGHTS_PATH), strict=False)
            print('加载权重成功：', WEIGHTS_PATH)
        except Exception as e:
            print('加载权重出现问题：', e)
        self.model.eval()

    def preprocess_image(self, pil_img):
        """把 PIL Image 变为模型输入 tensor（1,3,H,W）"""
        img = np.array(pil_img.convert('RGB'))
        img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - NORMALIZE_MEAN) / NORMALIZE_STD
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        return img_tensor, img  # 返回张量与原始 numpy resized 图

    def detect(self, pil_img):
        """前向推理，返回分割 mask (H,W) 的 numpy 数组（值为类别 id）
        注意：此函数内部用于演示，真实行为依赖于模型的 output 格式
        """
        x_tensor, resized_np = self.preprocess_image(pil_img)
        with torch.no_grad():
            output = self.model(x_tensor)
        mask_img = post_process(output, filter=False, threshold=0.05)  # 取你提供的 post_process
        mask_np = mask_img[0].cpu().numpy().astype(np.uint8)
        # 另把 output 保存到实例，以便后续读取 offsets
        self.last_output = output
        self.last_resized = resized_np
        return mask_np, output

    def get_deform_offsets_for_level(self, layer_id, lvl_id):
        """
        从模型内部读取 offsets 并 reshape 成 [B, H, W, n_head, n_lvl, n_points, 2]
        返回 torch.Tensor（在 CPU 上）
        """
        # 直接取你原始写法里对应的路径
        offsets = self.model.sem_seg_head.pixel_decoder.transformer.encoder.layers[layer_id].self_attn.offsets
        # offsets 原始形状假设为 [B, L, n_head, n_lvl, n_points, 2]
        # split 到各个 feature map
        split_size_or_sections = [i[0] * i[1] for i in FEATUREMAP_SZ]
        offsets_splits = torch.split(offsets, split_size_or_sections, dim=1)
        offsets_lvl = offsets_splits[lvl_id]  # [B, L_lvl, n_head, n_lvl, n_points, 2]
        b, _, n_head, n_lvl, n_points, c2 = offsets_lvl.shape
        H, W = FEATUREMAP_SZ[lvl_id]
        # reshape 成 [B, H, W, n_head, n_lvl, n_points, 2]
        offsets_lvl = offsets_lvl.reshape(b, H, W, n_head, n_lvl, n_points, c2)
        return offsets_lvl.cpu()

    def get_sampling_points_at(self, click_xy, layer_id, lvl_id):
        """
        click_xy: 在 resized 图像（IMG_SIZE）中的像素坐标 (x, y) （整数）
        返回：list of sampling points per head & per lvl
          返回字典：{
            'base_feat_coord': (fy, fx), # 在 feature map 上的整数坐标（h, w）
            'samples': np.array shape [n_head, n_lvl, n_points, 2] （以原图像坐标为单位, float）
          }
        说明：offset 的单位/含义在不同实现中可能不同。常见两种情况：
          (A) offsets 表示相对于 feature-grid 的像素偏移（单位为 feature map 像素）
          (B) offsets 表示归一化偏移（-1..1 或 以 patch 大小为单位）
        本函数会尝试自动检测并将采样点最终转换为原始 resized 图像坐标。
        """
        # 先把 click 转为对应 feature map 上的坐标
        img_h, img_w = IMG_SIZE
        fx = int(click_xy[0])
        fy = int(click_xy[1])
        # 计算 feature map 尺寸
        feat_H, feat_W = FEATUREMAP_SZ[lvl_id]
        # 将原图像坐标映射到 feature map 像素坐标（浮点）
        # 注意：x 对应 width 方向
        feat_x = (fx / float(img_w)) * feat_W
        feat_y = (fy / float(img_h)) * feat_H
        # 取最近的整数像素索引
        idx_x = min(max(int(round(feat_x)), 0), feat_W - 1)
        idx_y = min(max(int(round(feat_y)), 0), feat_H - 1)

        offsets_lvl = self.get_deform_offsets_for_level(layer_id, lvl_id)  # [B, H, W, n_head, n_lvl, n_points, 2]
        # 只支持 batch=1
        offsets_at = offsets_lvl[0, idx_y, idx_x]  # [n_head, n_lvl, n_points, 2]
        offsets_np = offsets_at.numpy()  # numpy
        n_head, n_lvl, n_points, _ = offsets_np.shape

        # 计算基准采样位置（grid），这里我们认为 base grid 就是 (idx_y, idx_x)
        # 不同 lvl 的采样点通常在该 lvl 的坐标系里，所以 base 坐标相同（idx_y, idx_x）
        base = np.array([idx_x, idx_y], dtype=np.float32)  # 注意顺序 (x, y)

        # 决定 offsets 的单位：如果偏移值的绝对最大值小于 2，我们认为它是归一化或相对小量 -> 需要缩放
        max_offset = np.max(np.abs(offsets_np))
        # 如果offset非常小（<2），尝试把它当作normalized并按 featuremap 尺度放大
        if max_offset < 2.5:
            print(111)
            # 假设 offsets 是相对于 featuremap 尺寸的 [-1,1] 或者相对 patch 的小数
            # 我们把它当作相对于 feat 尺度的相对位移：乘以 (feat_W, feat_H)
            # 这一步是启发式的：不同实现可能不一样，你可以根据实际模型调整
            scale_x = feat_W
            scale_y = feat_H
            offsets_np[..., 0] = offsets_np[..., 0] * scale_x
            offsets_np[..., 1] = offsets_np[..., 1] * scale_y
        # 现在 offsets_np 单位为 feature map 像素

        # sampling points in featuremap coordinates:
        base_b = base.reshape(1, 1, 1, 2)
        samp_feat = offsets_np + base_b
        # 把 featuremap 坐标映射回原图 resized 坐标
        samp_img = np.zeros_like(samp_feat)
        samp_img[..., 0] = (samp_feat[..., 0] / float(feat_W)) * img_w  # x
        samp_img[..., 1] = (samp_feat[..., 1] / float(feat_H)) * img_h  # y

        return {
            'base_feat_coord': (idx_y, idx_x),
            'samples_feat': samp_feat,  # featuremap coords
            'samples_img': samp_img,    # img coords (x,y)
            'n_head': n_head,
            'n_lvl': n_lvl,
            'n_points': n_points
        }


# --------------------------
# 复用你原先的 post_process 与 semantic_inference 函数（略作调整，放在这里便于导入）
# --------------------------
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

def semantic_inference(mask_cls, mask_pred):
    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
    mask_pred = mask_pred.sigmoid()
    semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
    return semseg


def post_process(output, filter, threshold=0.5):
    mask_cls_results = output["pred_logits"]
    mask_pred_results = output["pred_masks"]
    mask_pred_results = F.interpolate(
        mask_pred_results,
        scale_factor=cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
        mode="bilinear",
        align_corners=False,
    )
    pred_masks = semantic_inference(mask_cls_results, mask_pred_results)
    if filter:
        probs = torch.softmax(pred_masks, dim=1)
        conf, _ = torch.max(probs, dim=1)
        ignore_mask = conf < threshold
        mask_img = torch.argmax(pred_masks, dim=1)
        mask_img[ignore_mask] = 255
    else:
        mask_img = torch.argmax(pred_masks, dim=1)
    return mask_img


# --------------------------
# PyQt5 GUI 实现
# --------------------------
class ImageLabel(QtWidgets.QLabel):
    """用于显示图像并捕获鼠标点击坐标（坐标以 QLabel 显示尺寸为准）"""
    clicked = QtCore.pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(True)
        self.pix = None

    def setPixmap(self, pixmap: QtGui.QPixmap):
        super().setPixmap(pixmap)
        self.pix = pixmap

    def mousePressEvent(self, event):
        if self.pix is None:
            return
        # 计算点击在 pixmap 上的坐标（映射 QLabel 尺寸到 pixmap 像素）
        lbl_w = self.width()
        lbl_h = self.height()
        pm_w = self.pix.width()
        pm_h = self.pix.height()
        x = event.pos().x()
        y = event.pos().y()
        # 将 QLabel 的坐标映射到 pixmap 坐标
        px = int(x * (pm_w / lbl_w))
        py = int(y * (pm_h / lbl_h))
        self.clicked.emit(px, py)


class MainWindow(QtWidgets.QWidget):
    def __init__(self, model_wrapper):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.init_ui()
        self.current_pil = None
        self.resized_np = None
        self.last_mask = None
        self.annotated_img = None

    def init_ui(self):
        self.setWindowTitle('Deformable Attention Visualizer')
        self.resize(1200, 700)

        # 左侧原图，右侧掩码/结果
        self.left_img_label = ImageLabel(self)
        self.left_img_label.setFixedSize(480, 480)
        self.left_img_label.clicked.connect(self.on_click_image)

        self.right_img_label = QtWidgets.QLabel(self)
        self.right_img_label.setFixedSize(480, 480)
        self.right_img_label.setScaledContents(True)

        # 控件：文件名、加载按钮、detect 按钮、layer/lvl 选择、保存按钮
        self.filename_edit = QtWidgets.QLineEdit(self)
        self.browse_btn = QtWidgets.QPushButton('Browse', self)
        self.browse_btn.clicked.connect(self.on_browse)

        self.detect_btn = QtWidgets.QPushButton('Detect', self)
        self.detect_btn.clicked.connect(self.on_detect)

        self.layer_spin = QtWidgets.QSpinBox(self)
        self.layer_spin.setRange(0, 10)
        self.layer_spin.setValue(0)
        self.lvl_spin = QtWidgets.QSpinBox(self)
        self.lvl_spin.setRange(0, len(FEATUREMAP_SZ) - 1)
        self.lvl_spin.setValue(0)

        self.save_btn = QtWidgets.QPushButton('Save Annotated', self)
        self.save_btn.clicked.connect(self.on_save)

        # 说明文字
        self.info_text = QtWidgets.QLabel('点击左侧图像选择点，右侧显示当前预测掩码。', self)

        # 布局
        left_col = QtWidgets.QVBoxLayout()
        left_col.addWidget(self.left_img_label)
        left_col.addWidget(self.info_text)

        right_col = QtWidgets.QVBoxLayout()
        right_col.addWidget(self.right_img_label)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel('Image:'))
        controls.addWidget(self.filename_edit)
        controls.addWidget(self.browse_btn)
        controls.addWidget(self.detect_btn)
        controls.addWidget(QtWidgets.QLabel('layer_id'))
        controls.addWidget(self.layer_spin)
        controls.addWidget(QtWidgets.QLabel('lvl_id'))
        controls.addWidget(self.lvl_spin)
        controls.addWidget(self.save_btn)

        main_h = QtWidgets.QHBoxLayout()
        main_h.addLayout(left_col)
        main_h.addLayout(right_col)

        vmain = QtWidgets.QVBoxLayout(self)
        vmain.addLayout(controls)
        vmain.addLayout(main_h)

    def on_browse(self):
        # 简单从数据根目录里选择
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open image', DATA_ROOT, "Images (*.jpg *.png *.jpeg)")
        if fname:
            self.filename_edit.setText(fname)
            self.load_image_from_path(fname)

    def load_image_from_path(self, path):
        pil = Image.open(path).convert('RGB')
        self.current_pil = pil
        # resize preview 使用 IMG_SIZE，保持与推理一致
        resized = pil.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)
        self.resized_np = np.array(resized)
        qimg = QtGui.QImage(self.resized_np.data, self.resized_np.shape[1], self.resized_np.shape[0], self.resized_np.strides[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.left_img_label.width(), self.left_img_label.height(), QtCore.Qt.KeepAspectRatio)
        self.left_img_label.setPixmap(pix)
        # 清除右图
        self.right_img_label.clear()
        self.info_text.setText('已加载: %s' % os.path.basename(path))

    def on_detect(self):
        if self.current_pil is None:
            QtWidgets.QMessageBox.warning(self, 'Warning', '先选择图片再检测')
            return
        mask_np, output = self.model_wrapper.detect(self.current_pil)
        # mask_np 是 IMG_SIZE 下的类别图
        # 可视化 mask：用伪色放到右图
        mask_vis = self.make_mask_overlay(mask_np, self.resized_np)
        qimg = QtGui.QImage(mask_vis.data, mask_vis.shape[1], mask_vis.shape[0], mask_vis.strides[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.right_img_label.width(), self.right_img_label.height(), QtCore.Qt.KeepAspectRatio)
        self.right_img_label.setPixmap(pix)
        self.last_mask = mask_vis
        self.annotated_img = self.resized_np.copy()

    def make_mask_overlay(self, mask_np, img_np, alpha=0.6):
        VOC_COLORMAP = [
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
            [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
            [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
            [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 12]
        ]
        """
        mask_np: [H, W]，类别 id
        img_np: [H, W, 3]，原图（RGB）
        alpha: 透明度，0~1之间

        返回融合后的图像，np.uint8格式
        """
        h, w = mask_np.shape
        vis = img_np.copy().astype(np.float32)
        max_label = np.max(mask_np)

        for c in range(0, max_label + 1):
            mask_c = (mask_np == c)
            if c < len(VOC_COLORMAP):
                color = np.array(VOC_COLORMAP[c], dtype=np.float32)
            else:
                # 超出范围用黑色
                color = np.array([0, 0, 0], dtype=np.float32)

            # 对该类别区域做 alpha 融合
            vis[mask_c] = vis[mask_c] * (1 - alpha) + color * alpha

        vis = np.clip(vis, 0, 255).astype(np.uint8)
        return vis

    def on_click_image(self, px, py):
        # 先把显示坐标映射回图像坐标
        H_img, W_img = self.resized_np.shape[:2]
        W_label = self.left_img_label.width()
        H_label = self.left_img_label.height()

        mapped_x = int(px * W_img / W_label)
        mapped_y = int(py * H_img / H_label)

        # 后续所有用 mapped_x, mapped_y 替换原 px, py

        # 检查边界
        if mapped_x < 0 or mapped_x >= W_img or mapped_y < 0 or mapped_y >= H_img:
            self.info_text.setText('点击位置超出图像范围！')
            return

        # 下面替换 px, py 为 mapped_x, mapped_y
        layer_id = int(self.layer_spin.value())
        lvl_id = int(self.lvl_spin.value())

        try:
            info = self.model_wrapper.get_sampling_points_at((mapped_x, mapped_y), layer_id, lvl_id)
        except Exception as e:
            self.info_text.setText('读取 offsets 失败：%s' % str(e))
            return

        samp_img = info['samples_img']  # [n_head, n_lvl, n_points, 2]
        n_head = info['n_head']
        n_lvl = info['n_lvl']
        n_points = info['n_points']

        canvas_rgb = self.resized_np.copy()
        canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)

        # 用映射后的坐标画黑色中心点
        if 0 <= mapped_x < W_img and 0 <= mapped_y < H_img:
            cv2.circle(canvas_bgr, (mapped_x, mapped_y), 4, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

        # 画采样点（同之前）
        for hidx in range(n_head):
            color_rgb = DEFAULT_COLORS[hidx % len(DEFAULT_COLORS)]
            color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
            for lidx in range(n_lvl):
                size = max(1, 2 + (n_lvl - lidx - 1))
                for pidx in range(n_points):
                    x = int(round(samp_img[hidx, lidx, pidx, 0]))
                    y = int(round(samp_img[hidx, lidx, pidx, 1]))
                    if x < 0 or x >= W_img or y < 0 or y >= H_img:
                        continue
                    cv2.circle(canvas_bgr, (x, y), size, color_bgr, thickness=-1, lineType=cv2.LINE_AA)

        canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
        canvas_rgb = np.ascontiguousarray(canvas_rgb)

        qimg = QtGui.QImage(canvas_rgb.data, canvas_rgb.shape[1], canvas_rgb.shape[0],
                            canvas_rgb.strides[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(W_label, H_label, QtCore.Qt.KeepAspectRatio,
                                                   QtCore.Qt.SmoothTransformation)
        self.left_img_label.setPixmap(pix)
        self.annotated_img = canvas_rgb

        self.info_text.setText(f'显示 layer {layer_id} lvl {lvl_id}: head={n_head} points={n_points}')

    def on_save(self):
        if self.annotated_img is None:
            QtWidgets.QMessageBox.warning(self, 'Warning', '没有带注释的图像可保存')
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save annotated image', './annotated.png', 'PNG Files (*.png)')
        if fname:
            # success = cv2.imwrite(fname, cv2.cvtColor(self.annotated_img, cv2.COLOR_RGB2BGR))
            # if success:
            #     QtWidgets.QMessageBox.information(self, 'Saved', f'已保存至 {fname}')
            # else:
            #     QtWidgets.QMessageBox.warning(self, 'Failed', f'保存失败: {fname}')
            from PIL import Image
            img = Image.fromarray(self.annotated_img)
            img.save(fname)
            QtWidgets.QMessageBox.information(self, 'Saved', f'已保存至 {fname}')


# --------------------------
# 入口
# --------------------------
if __name__ == '__main__':
    cfg = get_args()
    # 构造模型包装器
    model_wrapper = ModelWrapper(cfg)

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(model_wrapper)
    win.show()
    sys.exit(app.exec_())
