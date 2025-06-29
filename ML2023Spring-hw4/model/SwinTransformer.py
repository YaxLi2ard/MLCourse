import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Tuple, Optional

class DepthwiseConv1d(nn.Module):
    """
    Depthwise卷积
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels 应该是 in_channels 的整数倍"

        # 使用 groups=in_channels，实现每个通道单独卷积
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)



class PointwiseConv1d(nn.Module):
    """
    Pointwise（逐点）卷积
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,  # 核大小为 1，即逐点卷积
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class Transpose(nn.Module):
    """
    封装 torch.transpose()，用于 nn.Sequential 中的维度交换。
    """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape  # 要交换的两个维度

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)  # 交换指定维度

class ConvProj(nn.Module):
    def __init__(self, in_dim, kernel_size, bias):
        super(ConvProj, self).__init__()
        self.conv_proj = nn.Sequential(
            Transpose(shape=(2, 1)),    # 转为 (B, C, T) 以适配 Conv1d
            DepthwiseConv1d(in_dim, in_dim, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),  # 深度卷积提取局部特征
            PointwiseConv1d(in_dim, in_dim, bias=bias),  # 通道扩展
            Transpose(shape=(2, 1)),    # 转会 (B, T, C)
        )

    def forward(self, x):
        return self.conv_proj(x)

class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module.
    Args:
        d_model: Embedding dimension.
        max_len: Maximum input length.
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor) -> None:
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : Input tensor B X T X C
        Returns:
            torch.Tensor: Encoded tensor B X T X C
        """
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb


class RelativeMultiHeadAttention(nn.Module):
    """
    使用相对位置编码的注意力模块
    """
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 8,
            dropout_p: float = 0.1,
    ):
        super(RelativeMultiHeadAttention, self).__init__()

        # 保证每个头的维度是整数
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)  # 用于缩放注意力分数

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
            query: Tensor,  # [b*nw, ws, d] 窗口注意力输入
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            attn_mask=None,  # [nw, ws, ws]，控制 shift 后不相邻的特征之间不注意
            padding_mask=None  # [b*nw, ws]，True 表示该位置为 padding，不参与注意力
    ) -> Tensor:
        batch_size = value.size(0)

        # 将 Q, K, V 和 位置嵌入 投影并 reshape 成 [batch, time, heads, head_dim]
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        # print(pos_embedding.shape)
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

        # 应用掩码，掩蔽无效位置
        # 加入 shift 后的 attention mask（防止不该注意的区域互相注意）
        if attn_mask is not None:
            # attn_mask: [B_, 1, N, N]
            score = score + attn_mask  # mask的地方为非常小的负值，和masked_fill原理相同

        # 加入 padding mask（屏蔽 pad 区域）
        if padding_mask is not None:
            # padding_mask: [B_, N] -> [B_, 1, 1, N]
            # 被 mask 的地方设置为非常小的负值
            score = score.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # Softmax 归一化得到注意力权重
        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        # 乘以 Value 得到上下文向量，并还原维度顺序
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        # 经过输出线性变换
        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        # 在最后一维前加一列0，用于偏移
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
        # reshape：将 time1 调整到最后维，再移除第一行
        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]
        return pos_score

def get_attn_mask_1d(seq_len, window_size, shift_size, device):
    """
    构建 shift 后的 attention mask，防止不同窗口的数据进行注意力计算。
    """
    # 构建帧索引标号
    img_mask = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T]
    if shift_size > 0:
        img_mask = torch.roll(img_mask, -shift_size, dims=1)
    # 分窗口
    num_windows = (seq_len + window_size - 1) // window_size
    img_mask = F.pad(img_mask, (0, num_windows * window_size - seq_len), value=-1)
    mask_windows = img_mask.view(1, num_windows, window_size)  # [1, num_windows, window_size]
    mask_windows = mask_windows.squeeze(0)  # [num_windows, window_size]

    # 对每个窗口，构造 pair-wise 不等的地方设置为 -inf
    attn_mask = (mask_windows.unsqueeze(2) != mask_windows.unsqueeze(1)).float() * -100.0
    return attn_mask  # [num_windows, window_size, window_size]

def get_padding_mask(lengths, max_len=None):
    """
    根据每个样本有效长度，构建 padding mask。
    """
    B = lengths.shape[0]
    if max_len is None:
        max_len = lengths.max().item()
    idxs = torch.arange(max_len).unsqueeze(0).expand(B, -1).to(lengths.device)
    mask = idxs >= lengths.unsqueeze(1)
    return mask  # [B, max_len]


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, shift_size, num_heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        # 相对位置编码模块
        # self.positional_encoding = PositionalEncoding(d_model)
        self.positional_encoding = RelPositionalEncoding(dim)
        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        # self.norm1 = nn.Sequential(
        #     Transpose(shape=(1, 2)),    # 转为 (B, C, T) 以适配 bn1d
        #     nn.BatchNorm1d(dim),  # 批归一化，加速收敛
        #     Transpose(shape=(1, 2)),
        # )
        self.attn = RelativeMultiHeadAttention(dim, num_heads, dropout)
        # Dropout 层，用于正则化
        self.dropout = nn.Dropout(p=dropout)

        self.attn_mask_shift = None
        self.attn_mask_noshift = None

    def forward(self, x):
        # x = x.transpose(1, 0)
        # print(type(x))            # 确认x类型
        # print(x.shape)            # 确认x的形状
        # print(type(x.shape))      # 打印shape的类型
        # B, T, C = x.shape
        B, T, C = [int(d) for d in x.shape]
        # print(B, type(B))         # 这里B应该是int
        # 获取padding和窗口移动造成的掩码
        padding_mask = torch.zeros([B, T], dtype=torch.bool).to(x.device)  # [B, T]
        # if self.training:
        #     if self.shift_size > 0:
        #         if self.attn_mask_shift is None:
        #             self.attn_mask_shift = get_attn_mask_1d(x.size(1), self.window_size, self.shift_size, x.device)
        #         attn_mask = self.attn_mask_shift
        #     else:
        #         if self.attn_mask_noshift is None:
        #             self.attn_mask_noshift = get_attn_mask_1d(x.size(1), self.window_size, self.shift_size, x.device)
        #         attn_mask = self.attn_mask_noshift
        # else:
        #     attn_mask = get_attn_mask_1d(x.size(1), self.window_size, self.shift_size, x.device)
        attn_mask = get_attn_mask_1d(x.size(1), self.window_size, self.shift_size, x.device)
        
        # 如果不能整除，需要 pad
        pad_len = (self.window_size - T % self.window_size) % self.window_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))  # pad 到窗口整除
            if padding_mask is not None:
                padding_mask = F.pad(padding_mask, (0, pad_len), value=True)

        # shift：先对序列循环移动
        if self.shift_size > 0:
            x = torch.roll(x, shifts=-self.shift_size, dims=1)
            if padding_mask is not None:
                padding_mask = torch.roll(padding_mask, shifts=-self.shift_size, dims=1)

        # 拆分成窗口：[B*num_windows, window_size, C]
        B_pad, T_pad, _ = x.shape
        num_windows = T_pad // self.window_size
        x_windows = x.view(B_pad, num_windows, self.window_size, C).reshape(-1, self.window_size, C)

        # 同样处理 padding_mask（[B, T] -> [B*num_windows, window_size]）
        if padding_mask is not None:
            mask_windows = padding_mask.view(B_pad, num_windows, self.window_size).reshape(-1, self.window_size)
        else:
            mask_windows = None
        
        # 获取序列长度对应的位置编码（形状为 [seq_len, d_model]）
        pos_embedding = self.positional_encoding(x_windows)
        # 扩展为 (batch_size, seq_len, d_model)
        pos_embedding = pos_embedding.repeat(x_windows.shape[0], 1, 1)
        
        # layer normalize
        x_windows = self.norm1(x_windows)
        
        # attention
        attn_mask = attn_mask.repeat_interleave(B, dim=0)  # [B * num_windows, window_size, window_size]
        attn_mask = attn_mask.unsqueeze(1)  # [B * num_windows, 1(num_heads dim), window_size, window_size]
        x_windows = self.attn(
            x_windows,   # query
            x_windows,   # key
            x_windows,   # value
            pos_embedding=pos_embedding,
            attn_mask=attn_mask,
            padding_mask=mask_windows
        )


        # 恢复原始 shape：[B, T, C]
        x = x_windows.view(B_pad, num_windows, self.window_size, C).reshape(B_pad, T_pad, C)

        # shift 复原
        if self.shift_size > 0:
            x = torch.roll(x, shifts=self.shift_size, dims=1)

        # 去除 padding 部分
        if pad_len > 0:
            x = x[:, :-pad_len, :]

        # 最后加 dropout
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1):
        super(SelfAttention, self).__init__()
        # 相对位置编码模块
        # self.positional_encoding = PositionalEncoding(d_model)
        self.positional_encoding = RelPositionalEncoding(d_model)
        # LayerNorm：用于预归一化
        self.layer_norm = nn.LayerNorm(d_model)
        # self.norm = nn.Sequential(
        #     Transpose(shape=(1, 2)),    # 转为 (B, C, T) 以适配 bn1d
        #     nn.BatchNorm1d(d_model),  # 批归一化，加速收敛
        #     Transpose(shape=(1, 2)),
        # )
        # 相对位置多头注意力模块
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        # Dropout 层，用于正则化
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        # 输入张量形状为 (batch_size, seq_len, d_model)
        batch_size, seq_length, _ = inputs.size()
        # 获取序列长度对应的位置编码（形状为 [seq_len, d_model]）
        # pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = self.positional_encoding(inputs)
        # 扩展为 (batch_size, seq_len, d_model)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)
        # 对输入做 LayerNorm 归一化
        inputs = self.layer_norm(inputs)
        # 执行相对位置多头注意力，传入 pos_embedding 和mask
        outputs = self.attention(
            inputs,   # query
            inputs,   # key
            inputs,   # value
            pos_embedding=pos_embedding,
        )
        # 最后加 dropout
        outputs = self.dropout(outputs)
        return outputs

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, is_window, window_size, shift_size, num_heads, mlp_ratio=1.0, dropout=0.1):
        """
        Swin Transformer Block，包含一个窗口注意力模块和一个前馈网络（MLP），并加入残差连接和 LayerNorm。
        """
        super().__init__()
        shift_size = window_size // 2
        if is_window:
            self.attn = WindowAttention(dim, window_size, shift_size, num_heads, dropout)
        else:
            self.attn = SelfAttention(dim, num_heads, dropout)
        # 第二个 LayerNorm，位于 MLP 之前
        self.norm2 = nn.LayerNorm(dim)
        # self.norm2 = nn.Sequential(
        #     Transpose(shape=(1, 2)),    # 转为 (B, C, T) 以适配 bn1d
        #     nn.BatchNorm1d(dim),  # 批归一化，加速收敛
        #     Transpose(shape=(1, 2)),
        # )
        # 前馈网络部分（MLP）
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 残差连接 1：窗口注意力前
        shortcut = x
        x = self.attn(x)
        x = x + shortcut  # 第一条残差连接
        # 残差连接 2：前馈网络前
        shortcut2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + shortcut2  # 第二条残差连接

        return x

class PatchMerging(nn.Module):
    def __init__(self, dim):
        """
        Patch Merging模块：将序列长度减半，维度扩大为2倍（[B, T, C] → [B, T//2, 2*C]）
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim * 2)
        self.reduction = nn.Linear(dim * 2, dim * 2)

    def forward(self, x):
        B, T, C = x.shape
        # 如果长度是奇数，则 pad 1 个时间步
        if T % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1))  # 在时间维度 pad 1
        # 将相邻两个 patch 合并：每两个时间步拼接通道
        x = x.view(B, -1, 2, C)  # [B, T//2, 2, C]
        x = x.reshape(B, -1, 2 * C)  # [B, T//2, 2*C]
        # LayerNorm + 线性映射
        x = self.norm(x)
        x = self.reduction(x)  # [B, T//2, 2*C]
        return x


class SwinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 256
        n_spks = 600
        dropout = 0.1
        num_heads = 2
        # window_size = 32
        # shift_size = window_size // 2
        # 投影到 d_model 维度
        self.prenet = nn.Linear(40, d_model)


        self.encoder = nn.Sequential(
            SwinTransformerBlock(d_model, is_window=True, window_size=32, shift_size=0, num_heads=num_heads, dropout=dropout),
            SwinTransformerBlock(d_model, is_window=True, window_size=32, shift_size=32//2, num_heads=num_heads, dropout=dropout),

            # SwinTransformerBlock(d_model, is_window=True, window_size=32, shift_size=0, num_heads=num_heads, dropout=dropout),
            # SwinTransformerBlock(d_model, is_window=True, window_size=32, shift_size=32//2, num_heads=num_heads, dropout=dropout),

            # SwinTransformerBlock(d_model, window_size=32, shift_size=0, num_heads=num_heads, dropout=dropout),
            # SwinTransformerBlock(d_model, window_size=32, shift_size=32//2, num_heads=num_heads, dropout=dropout),
            
            # PatchMerging(d_model),
            SwinTransformerBlock(d_model, is_window=True, window_size=64, shift_size=0, num_heads=num_heads, dropout=dropout),
            SwinTransformerBlock(d_model, is_window=True, window_size=64, shift_size=64//2, num_heads=num_heads, dropout=dropout),
            
            SwinTransformerBlock(d_model, is_window=True, window_size=64, shift_size=0, num_heads=num_heads, dropout=dropout),
            SwinTransformerBlock(d_model, is_window=True, window_size=64, shift_size=64//2, num_heads=num_heads, dropout=dropout),

            # SwinTransformerBlock(d_model, is_window=True, window_size=64, shift_size=0, num_heads=num_heads, dropout=dropout),
            # SwinTransformerBlock(d_model, is_window=True, window_size=64, shift_size=64//2, num_heads=num_heads, dropout=dropout),

            SwinTransformerBlock(d_model, is_window=False, window_size=0, shift_size=0, num_heads=num_heads, dropout=dropout),
            # SwinTransformerBlock(d_model * 2, window_size, shift_size=0, num_heads=num_heads, dropout=dropout),
            # SwinTransformerBlock(d_model * 2, window_size, shift_size=window_size//2, num_heads=num_heads, dropout=dropout),
        )

        # 分类层（输入维度需要改为 2 * d_model）
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_spks),
        )

    def forward(self, mels):
        x = self.prenet(mels)          # [B, T, d_model]

        x = self.encoder(x)   

        stats = x.mean(dim=1)     
        out = self.pred_layer(stats) 
        return out



if __name__ == '__main__':
    model = SwinConformer()
    x = torch.rand(3, 128, 40)
    y = model(x)
    print(y.shape)