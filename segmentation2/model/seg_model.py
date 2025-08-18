import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.p2t import p2t_base, p2t_large
from .backbone.swin import swin_base
from .backbone.vit import vit_b_16, vit_b_16_office
from .fpn import FPNHead, FPNHead_
from .maskformer import MultiScaleMaskedTransformerDecoder
import torchvision.models as tv_models
from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table
import timm
import re

class SegModel(nn.Module):
    def __init__(self, backbone='p2t', head='fpn', num_cls=21, pretrained=True):
        super().__init__()
        self.backbone_type = backbone
        self.head_type = head
        # backbone
        assert self.backbone_type in ['p2t', 'swin', 'vit'], f'backbone {self.backbone_type} not supported'
        if self.backbone_type == 'p2t':
            self.backbone = p2t_base()
            out_dim = self.backbone.embed_dims
        elif self.backbone_type == 'swin':
            self.backbone = swin_base()
            out_dim = [128, 256, 512, 1024]
        elif self.backbone_type == 'vit':
            self.backbone = vit_b_16_office()
            self.proj_memory = nn.Conv2d(768, 256, 1, padding=0, bias=False)
            self.proj_feat = nn.Conv2d(768, 256, 1, padding=0, bias=False)
            out_dim = [256, 256, 256, 256]
        else:
            raise NotImplementedError
        if pretrained:
            self.load_backbone_weight()
        # head
        assert self.head_type in ['fpn', 'maskformer'], f'head {self.head_type} not supported'
        self.fpn = FPNHead_(in_channels=out_dim, channels=256, kernel_size=3)
        if self.head_type == 'fpn':
            self.classifier = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_cls, 1)
            )
        elif self.head_type == 'maskformer':
            # 混合fpn的结果
            self.conv1_1 = nn.Conv2d(256, 256, kernel_size=1)
            # 中间特征通道变为一致
            self.latent = nn.ModuleList()
            for dim in out_dim[1:]:
                self.latent.append(
                    nn.Conv2d(dim, 256, kernel_size=1)
                )
            num_lvl = 1 if self.backbone_type=='vit' else 3
            self.decoder = MultiScaleMaskedTransformerDecoder(num_classes=num_cls, num_feature_levels=num_lvl)

        else:
            raise NotImplementedError

    
    def forward(self, x):
        # output = {}
        # feats = self.backbone(x)
        # feat_fpn = self.fpn(feats)
        # if self.head_type == 'fpn':  # 直接用fpn特征分类
        #     logits = self.classifier(feat_fpn)
        #     output['logits'] = logits
        # elif self.head_type == 'maskformer':
        #     feats_latent = []
        #     for i, feat in enumerate(feats[1:]):
        #         feats_latent.append(self.latent[i](feat))
        #     feat_fpn = self.conv1_1(feat_fpn)
        #     decoder_output = self.decoder(feats_latent, feat_fpn)
        #     output.update(decoder_output)
        # else:
        #     raise NotImplementedError
        # return output
        if self.backbone_type in ['swin', 'p2t']:
            output = {}
            feats = self.backbone(x)
            feats_fpn = self.fpn(feats)
            if self.head_type == 'fpn':  # 直接用fpn特征分类
                logits = self.classifier(feats_fpn[0])
                output['logits'] = logits
            elif self.head_type == 'maskformer':
                decoder_output = self.decoder(feats_fpn[1:], feats_fpn[0])
                output.update(decoder_output)
            else:
                raise NotImplementedError
            return output
        elif self.backbone_type in ['vit']:
            output = {}
            feat, memory = self.backbone.forward_semantic(x)
            if self.head_type == 'fpn':
                logits = self.proj_feat(feat)
                output['logits'] = logits
            elif self.head_type == 'maskformer':
                mask_feat = self.proj_feat(feat)
                memory = self.proj_memory(memory)
                decoder_output = self.decoder([memory], mask_feat)
                output.update(decoder_output)
            return output
            

    
    def load_backbone_weight(self):
        if self.backbone_type == 'p2t':
            self.backbone = load_p2t(self.backbone, 'base')
        elif self.backbone_type == 'swin':
            self.backbone = load_swin(self.backbone, 'base')
        elif self.backbone_type == 'vit':
            self.backbone = load_vit(self.backbone, 'base')
        else:
            raise NotImplementedError


def load_vit(model, type):
    model_pretrained = tv_models.vit_b_16(weights="IMAGENET1K_V1")
    pretrained_state_dict = model_pretrained.state_dict()
    load_info = model.load_state_dict(pretrained_state_dict, strict=False)
    print_load_info(load_info)
    return model

def load_p2t(model, type):
    checkpoint = torch.load(f'./model/backbone/p2t_{type}.pth')
    load_info = model.load_state_dict(checkpoint, strict=False)
    print_load_info(load_info)
    return model

def load_swin(model, type):
    # swin = timm.create_model(f'swin_{type}_patch4_window7_224', pretrained=True)
    swin = timm.create_model(f'swin_{type}_patch4_window7_224.ms_in22k_ft_in1k', pretrained=True)
    pretrained_state_dict = swin.state_dict()
    new_state_dict = {}
    # timm 的 swin 和 使用的 swin 中的 patch merging 索引编号错位了，因为timm里的patch merging 分给了当前 layer，而标准实现是分给了下一个layer
    for k, v in pretrained_state_dict.items():
        import re
        m = re.match(r'(layers)\.(\d+)\.(downsample\..+)', k)
        if m:
            prefix, idx_str, suffix = m.groups()
            idx = int(idx_str) - 1  # 索引+1
            new_key = f"{prefix}.{idx}.{suffix}"
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    load_info = model.load_state_dict(new_state_dict, strict=False)
    print_load_info(load_info)
    return model

def print_load_info(load_info):
    print("❌ 未加载成功的参数（模型需要但权重中没有）:")
    print(load_info.missing_keys)
    # 打印多余的参数（权重中有但模型中没有）
    print("⚠️ 多余的参数（权重中存在但模型中没有用到）:")
    print(load_info.unexpected_keys)
    print("Loaded weights successfully.")


