import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = 40
        num_cls = 600
        h_dim = 100
        # 特征投影层
        self.fc_pre = nn.Linear(in_dim, h_dim)
        # 单层 Transformer 编码器
        # self.encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=h_dim, dim_feedforward=h_dim * 2, nhead=1, dropout=0.1, batch_first=True
        # )
        self.encoder_layer = TransformerEncoderLayer(
            d_model=h_dim, dim_feedforward=h_dim * 2, nhead=1, dropout=0.1
        )
        # self.encoder_layer = RelTransformerEncoderLayer(
        #     d_model=h_dim, dim_feedforward=h_dim * 2, nhead=1, dropout=0.1
        # )
        # 分类器
        self.fc_cls = nn.Sequential(
            nn.Linear(h_dim, 256),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_cls),
        )

    def forward(self, mels):
        # mels: [b, t, 40]
        x = self.fc_pre(mels)         # [b, t, h_dim]
        x = self.encoder_layer(x)     # [b, t, h_dim]
        x = x.mean(dim=1)             # [b, h_dim]
        out = self.fc_cls(x)          # [b, num_cls]
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead=1, dim_feedforward=512, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        # 多头自注意力机制，这里使用 PyTorch 自带的实现
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)

        # 前馈神经网络部分：两层全连接+ReLU
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        # 残差连接和 LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        pos_embedding = self.positional_encoding(src.shape[1])
        src = src + pos_embedding
        # 第一步：多头自注意力 + 残差连接 + LayerNorm
        attn_output, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        # 第二步：前馈网络 + 残差连接 + LayerNorm
        ff_output = self.feedforward(src)
        src = src + self.dropout(ff_output)
        src = self.norm2(src)

        return src


class PositionalEncoding(nn.Module):
    """
    位置编码模块，来自论文 "Attention Is All You Need"。
    PE(pos, 2i)   = sin(pos / (10000^(2i / d_model)))
    PE(pos, 2i+1) = cos(pos / (10000^(2i / d_model)))
    """

    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        # 初始化一个大小为 [max_len, d_model] 的位置编码矩阵，禁止梯度更新
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        # 位置序列 [0, 1, ..., max_len-1]，形状为 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 指数因子（不同频率），形状为 [d_model/2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # 偶数维度使用正弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维度使用余弦函数
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加 batch 维度，变为 [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        # 注册为 buffer，随着模型移动device
        self.register_buffer('pe', pe)

    def forward(self, length: int):
        return self.pe[:, :length]

class RelativeMultiHeadAttention(nn.Module):
    """
    带有相对位置编码的多头注意力机制（来自 Transformer-XL）。
    传统的 Transformer 使用绝对位置编码，但 Transformer-XL 引入了相对位置编码，
    有助于模型更好地捕捉长距离依赖信息。
    """

    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
    ):
        super(RelativeMultiHeadAttention, self).__init__()

        # 保证每个头的维度是整数
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)  # 用于缩放注意力分数

        # QKV 和 位置嵌入线性变换
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)  # 位置编码不使用偏置

        # dropout 层
        self.dropout = nn.Dropout(p=dropout_p)

        # 可学习的偏置 u 和 v，用于相对位置注意力中的内容偏置和位置偏置
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))

        # 使用 Xavier 初始化偏置项
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        # 输出线性层
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
            self,
            query,
            key,
            value,
            pos_embedding,
            mask = None,
    ):
        batch_size = value.size(0)
        # 将 Q, K, V 和 位置嵌入 投影并 reshape 成 [batch, time, heads, head_dim]
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)
        # print(pos_embedding.shape)
        # print(query.shape)
        # 内容相关注意力分数: (Q + u) @ K^T
        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
    
        # 位置相关注意力分数: (Q + v) @ P^T
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)  # 调整位置信息对齐

        # 计算总注意力分数并缩放
        score = (content_score + pos_score) / self.sqrt_dim

        # 应用掩码，掩蔽无效位置（如 padding 或未来时刻）
        if mask is not None:
            mask = mask.unsqueeze(1)  # 使其适配多头维度
            score.masked_fill_(mask, -1e9)  # 设为极小值以避免被注意到

        # Softmax 归一化得到注意力权重
        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        # 乘以 Value 得到上下文向量，并还原维度顺序
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        # 经过输出线性变换
        return self.out_proj(context)

    def _relative_shift(self, pos_score):
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        # 在最后一维前加一列0，用于偏移
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
        # reshape：将 time1 调整到最后维，再移除第一行
        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]
        return pos_score

class RelTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead=1, dim_feedforward=512, dropout=0.1):
        super(RelTransformerEncoderLayer, self).__init__()
        self.positional_encoding = RelPositionalEncoding(d_model, max_pos_len=65)
        # 多头自注意力机制，这里使用 PyTorch 自带的实现
        self.self_attn = RelativeMultiHeadAttention(d_model= d_model, num_heads = nhead, dropout_p = dropout)

        # 前馈神经网络部分：两层全连接+ReLU
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        # 残差连接和 LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        pos_embedding = self.positional_encoding(src)
        pos_embedding = pos_embedding.repeat(src.shape[0], 1, 1)
        # 第一步：多头自注意力 + 残差连接 + LayerNorm
        attn_output = self.self_attn(
            src, src, src,
            pos_embedding
        )
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        # 第二步：前馈网络 + 残差连接 + LayerNorm
        ff_output = self.feedforward(src)
        src = src + self.dropout(ff_output)
        src = self.norm2(src)

        return src

class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module.
    Args:
        d_model: Embedding dimension.
        max_len: Maximum input length.
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000, max_pos_len=32) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_pos_len = max_pos_len
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x) -> None:
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x):
        """
        Args:
            x : Input tensor B X T X C
        Returns:
            torch.Tensor: Relative positional encoding B X (2T-1) X C
        """
        B, T, _ = x.size()
        self.extend_pe(x)

        center = self.pe.size(1) // 2
        full_len = 2 * T - 1
        max_len = 2 * self.max_pos_len - 1

        if full_len <= max_len:
            # 从中间切出 full_len 长度
            start = center - (full_len // 2)
            end = start + full_len
            pos_emb = self.pe[:, start:end, :]  # shape: [1, 2T-1, C]
        else:
            # 截取中间 max_len 长度，再填补两边
            start = center - (max_len // 2)
            end = start + max_len
            truncated = self.pe[:, start:end, :]  # shape: [1, max_len, C]

            pad_left = full_len // 2 - max_len // 2
            pad_right = full_len - max_len - pad_left

            left_pad = truncated[:, 0:1, :].expand(1, pad_left, -1)
            right_pad = truncated[:, -1:, :].expand(1, pad_right, -1)
            pos_emb = torch.cat([left_pad, truncated, right_pad], dim=1)  # [1, 2T-1, C]

        return pos_emb

if __name__ == '__main__':
    model = Transformer()
    x = torch.rand(3, 128, 40)
    y = model(x)
    print(y.shape)