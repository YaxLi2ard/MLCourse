import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config['num_heads']
        self.attention_head_size = int(config['hidden_size'] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.query = Linear(config['hidden_size'], self.all_head_size)
        # self.key = Linear(config['hidden_size'], self.all_head_size)
        # self.value = Linear(config['hidden_size'], self.all_head_size)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * self.all_head_size, config['hidden_size']))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * self.all_head_size))

        self.out_proj = Linear(config['hidden_size'], config['hidden_size'])
        self.attn_dropout = Dropout(config['attention_dropout_rate'])
        self.proj_dropout = Dropout(config['attention_dropout_rate'])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # mixed_query_layer = self.query(hidden_states)
        # mixed_key_layer = self.key(hidden_states)
        # mixed_value_layer = self.value(hidden_states)
        
        # query_layer = self.transpose_for_scores(mixed_query_layer)
        # key_layer = self.transpose_for_scores(mixed_key_layer)
        # value_layer = self.transpose_for_scores(mixed_value_layer)

        qkv = F.linear(hidden_states, self.in_proj_weight, self.in_proj_bias)  # [b, n, h]
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, self.num_attention_heads, self.attention_head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4) 
        query_layer, key_layer, value_layer = qkv[0], qkv[1], qkv[2]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out_proj(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config['hidden_size'], config['mlp_dim'])
        self.fc2 = Linear(config['mlp_dim'], config['hidden_size'])
        self.dropout = Dropout(config['dropout_rate'])

        self._init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = config['img_size']
        patch_size = config['patch_size']
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels, config['hidden_size'], patch_size, patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config['hidden_size']))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config['hidden_size']))

        self.dropout = Dropout(config['dropout_rate'])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln_1 = LayerNorm(config['hidden_size'], eps=1e-6)
        # self.self_attention = Attention(config)
        self.self_attention = nn.MultiheadAttention(config['hidden_size'], config['num_heads'], dropout=config['attention_dropout_rate'], batch_first=True)
        self.ln_2 = LayerNorm(config['hidden_size'], eps=1e-6)
        # self.mlp = Mlp(config)
        self.mlp = nn.Sequential(
            Linear(config['hidden_size'], config['mlp_dim']),
            nn.GELU(),
            Dropout(config['dropout_rate']),
            Linear(config['mlp_dim'], config['hidden_size']),
            Dropout(config['dropout_rate']),
        )

    def forward(self, x):
        h = x
        x = self.ln_1(x)
        # x = self.self_attention(x)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = x + h

        h = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = x + h
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        # self.encoder_norm = LayerNorm(config['hidden_size'], eps=1e-6)
        self.num_layers = config['num_layers']
        for i in range(self.num_layers):
            layer = Block(config)
            setattr(self, f"encoder_layer_{i}", layer)

    def forward(self, hidden_states):
        for i in range(self.num_layers):
            layer_block = getattr(self, f"encoder_layer_{i}")
            hidden_states = layer_block(hidden_states)
        # encoded = self.encoder_norm(hidden_states)
        encoded = hidden_states
        return encoded


# class Transformer(nn.Module):
#     def __init__(self, config):
#         super(Transformer, self).__init__()
#         img_size = config['img_size']
#         patch_size = config['patch_size']
#         n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
#         self.pos_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config['hidden_size']))
#         self.dropout = Dropout(config['dropout_rate'])
        
#         # self.embeddings = Embeddings(config)
#         self.layers = Encoder(config)

#         self.ln = LayerNorm(config['hidden_size'], eps=1e-6)

#     def forward(self, x):
#         # + 位置编码
#         embedding_output = x + self.pos_embeddings
#         embedding_output = self.dropout(embedding_output)
        
#         # embedding_output = self.embeddings(x)
#         encoded = self.layers(embedding_output)
        
#         encoded = self.ln(encoded)
#         return encoded


# class VisionTransformer(nn.Module):
#     def __init__(self, config):
#         super(VisionTransformer, self).__init__()
#         patch_size = config['patch_size']
#         self.conv_proj = Conv2d(3, config['hidden_size'], patch_size, patch_size)
#         self.class_token = nn.Parameter(torch.zeros(1, 1, config['hidden_size']))
        
#         self.num_classes = config['num_classes']

#         self.encoder = Transformer(config)
#         self.head = Linear(config['hidden_size'], self.num_classes)

#     def forward(self, x):
#         # patch embd
#         B = x.shape[0]
#         cls_tokens = self.class_token.expand(B, -1, -1)
#         x = self.conv_proj(x)  # [b, c, h, w]
#         x = x.flatten(2)  # [b, c, n]
#         x = x.transpose(-1, -2)  # [b, n, c]
#         x = torch.cat((cls_tokens, x), dim=1)  # [b, n+1, c]
        
#         x = self.encoder(x)
#         logits = self.head(x[:, 0])
#         return logits


# 二维形式的pos_embd
class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.layers = Encoder(config)
        self.ln = LayerNorm(config['hidden_size'], eps=1e-6)

    def forward(self, x):
        encoded = self.layers(x)
        encoded = self.ln(encoded)
        return encoded


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super(VisionTransformer, self).__init__()
        patch_size = config['patch_size']
        self.conv_proj = Conv2d(3, config['hidden_size'], patch_size, patch_size)
        self.pos_embd = nn.Parameter(torch.zeros(1, config['hidden_size'], config['img_size'][0], config['img_size'][1]))
        self.class_token = nn.Parameter(torch.zeros(1, 1, config['hidden_size']))
        self.num_classes = config['num_classes']
        self.dropout = Dropout(config['dropout_rate'])
        self.encoder = Transformer(config)
        self.head = Linear(config['hidden_size'], self.num_classes)

    def forward(self, x):
        # patch embd
        B = x.shape[0]
        cls_tokens = self.class_token.expand(B, -1, -1)
        x = self.conv_proj(x)  # [b, c, h, w]
        # + pos_embd
        x = x + F.interpolate(self.pos_embd, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        x = x.flatten(2)  # [b, c, n]
        x = x.transpose(-1, -2)  # [b, n, c]
        x = torch.cat((cls_tokens, x), dim=1)  # [b, n+1, c]
        
        x = self.encoder(x)
        logits = self.head(x[:, 0])
        return logits


class vit_b_16(VisionTransformer):
    def __init__(self, **kwargs):
        config = {
            'img_size': [320, 320],
            'num_classes': 10,
            'patch_size': [16, 16],
            'hidden_size': 768,
            'mlp_dim': 3072,
            'num_heads': 12,
            'num_layers': 12,
            'dropout_rate': 0.1,
            'attention_dropout_rate': 0.0
        }
        super().__init__(config)
        self.dropout = nn.Dropout(0.1)
        # self.interp_layer = Block(config)
        self.conv1 = nn.Conv2d(768, 768, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(768, 768, 3, padding=1, bias=False)

    def forward_feat(self, x):
        # patch embd
        B = x.shape[0]
        cls_tokens = self.class_token.expand(B, -1, -1)
        x = self.conv_proj(x)  # [b, c, h, w]
        self.ph, self.pw = x.shape[2], x.shape[3]
        # + pos_embd
        x = x + F.interpolate(self.pos_embd, size=(self.ph, self.pw), mode='bilinear', align_corners=False)
        x = x.flatten(2)  # [b, c, n]
        x = x.transpose(-1, -2)  # [b, n, c]
        x = torch.cat((cls_tokens, x), dim=1)  # [b, n+1, c]
        x = self.dropout(x)
        x = self.encoder(x)  # [b, n+1, c]
        return x

    # def forward_semantic(self, x):
    #     x = self.forward_feat(x)
    #     x = x[:, 1:, :]  # [b, n, c]
    #     x = x.transpose(2, 1)  # [b, c, n]
    #     x = x.view(x.shape[0], x.shape[1], self.ph, self.pw)  # [b, c, h, w]
    #     memory = x
    #     x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    #     x = x.flatten(2)  # [b, c, n]
    #     x = x.transpose(2, 1)  # [b, n, c]
    #     x = self.interp_layer(x)  # [b, n, c]
    #     x = x.transpose(2, 1)  # [b, c, n]
    #     x = x.view(x.shape[0], x.shape[1], self.ph*2, self.pw*2)
    #     x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    #     return x, memory

    def forward_semantic(self, x):
        x = self.forward_feat(x)
        x = x[:, 1:, :]  # [b, n, c]
        x = x.transpose(2, 1)  # [b, c, n]
        x = x.view(x.shape[0], x.shape[1], self.ph, self.pw)  # [b, c, h, w]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        memory = x
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2(x)
        return x, memory


class vit_b_16_office(nn.Module):
    def __init__(self):
        super().__init__()
        config = {
            'img_size': [320, 320],
            'num_classes': 10,
            'patch_size': [16, 16],
            'hidden_size': 768,
            'mlp_dim': 3072,
            'num_heads': 12,
            'num_layers': 12,
            'dropout_rate': 0.1,
            'attention_dropout_rate': 0.0
        }
        self.model = models.vit_b_16(weights="IMAGENET1K_V1")
        self.pos_embd = nn.Parameter(torch.zeros(1, 768, 320, 320))
        # self.interp_layer = Block(config)
        # self.interp_layer = nn.Conv2d(768, 768, 3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(768, 768, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(768, 768, 3, padding=1, bias=False)

    def forward_feat(self, x):
        # patch embd
        B = x.shape[0]
        cls_tokens = self.model.class_token.expand(B, -1, -1)
        x = self.model.conv_proj(x)  # [b, c, h, w]
        self.ph, self.pw = x.shape[2], x.shape[3]
        # + pos_embd
        x = x + F.interpolate(self.pos_embd, size=(self.ph, self.pw), mode='bilinear', align_corners=False)
        x = x.flatten(2)  # [b, c, n]
        x = x.transpose(-1, -2)  # [b, n, c]
        x = torch.cat((cls_tokens, x), dim=1)  # [b, n+1, c]
        x = self.model.encoder.ln(self.model.encoder.layers(self.model.encoder.dropout(x)))
        return x

    # def forward_semantic(self, x):
    #     x = self.forward_feat(x)
    #     x = x[:, 1:, :]  # [b, n, c]
    #     x = x.transpose(2, 1)  # [b, c, n]
    #     x = x.view(x.shape[0], x.shape[1], self.ph, self.pw)  # [b, c, h, w]
    #     memory = x
    #     x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    #     # x = x.flatten(2)  # [b, c, n]
    #     # x = x.transpose(2, 1)  # [b, n, c]
    #     x = self.interp_layer(x)  # [b, n, c]
    #     # x = x.transpose(2, 1)  # [b, c, n]
    #     # x = x.view(x.shape[0], x.shape[1], self.ph*2, self.pw*2)
    #     x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    #     return x, memory

    def forward_semantic(self, x):
        x = self.forward_feat(x)
        x = x[:, 1:, :]  # [b, n, c]
        x = x.transpose(2, 1)  # [b, c, n]
        x = x.view(x.shape[0], x.shape[1], self.ph, self.pw)  # [b, c, h, w]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        memory = x
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2(x)
        return x, memory

if __name__ == '__main__':
    model = vit_b_16()

    model_pretrained = models.vit_b_16(weights="IMAGENET1K_V1")
    pretrained_state_dict = model_pretrained.state_dict()
    load_info = model.load_state_dict(pretrained_state_dict, strict=False)
    print("❌ 未加载成功的参数（模型需要但权重中没有）:")
    print(load_info.missing_keys)
    # 打印多余的参数（权重中有但模型中没有）
    print("⚠️ 多余的参数（权重中存在但模型中没有用到）:")
    print(load_info.unexpected_keys)
    print("Loaded weights successfully.")

    x = torch.rand(1, 3, 320, 320)
    y = model(x)
    print(y.shape)