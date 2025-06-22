import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Tuple, Optional

class ResidualConnectionModule(nn.Module):
    """
    残差连接模块。
    输出 = 模块输出 × module_factor + 输入 × input_factor
    用于残差结构中控制主路径和残差路径的权重。
    """
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module  # 主路径模块
        self.module_factor = module_factor  # 主路径输出系数
        self.input_factor = input_factor    # 输入残差路径系数

    def forward(self, inputs: Tensor) -> Tensor:
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)


class Linear(nn.Module):
    """
    封装的线性层。
    使用 Xavier 均匀初始化权重，偏置初始化为 0。
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)  # Xavier 均匀初始化
        if bias:
            init.zeros_(self.linear.bias)  # 偏置初始化为 0

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class View(nn.Module):
    """
    封装 torch.view()，用于 nn.Sequential 中的 reshape 操作。
    """
    def __init__(self, shape: tuple, contiguous: bool = False):
        super(View, self).__init__()
        self.shape = shape              # 目标形状
        self.contiguous = contiguous    # 是否确保连续内存

    def forward(self, x: Tensor) -> Tensor:
        if self.contiguous:
            x = x.contiguous()  # 如果指定，先转为连续内存
        return x.view(*self.shape)  # 重新 reshape


class Transpose(nn.Module):
    """
    封装 torch.transpose()，用于 nn.Sequential 中的维度交换。
    """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape  # 要交换的两个维度

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)  # 交换指定维度


class Swish(nn.Module):
    """
    Swish 激活函数（由 Google 提出），形式为：x * sigmoid(x)
    比 ReLU 更平滑，性能更优，适用于深度网络中的各种任务。
    """
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    """
    Gated Linear Unit（门控线性单元）。
    最早在自然语言处理任务中提出（论文："Language Modeling with Gated Convolutional Networks"）。
    将输入按指定维度一分为二，前一半为主输出，后一半为门控，经过 sigmoid 激活后与前半相乘。
    """
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim  # 按该维度拆分输入张量

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)  # 拆成两半
        return outputs * gate.sigmoid()  # 门控机制



class FeedForwardModule(nn.Module):
    """
    Conformer 中的前馈模块（Feed Forward Module）。
    它采用了 pre-norm 残差结构，即在残差连接之前对输入做 LayerNorm 归一化。
    同时使用了 Swish 激活函数和 Dropout 进行正则化，有助于提升网络的泛化能力。
    """

    def __init__(
        self,
        encoder_dim: int = 512,            # 输入的特征维度
        expansion_factor: int = 4,         # 扩展因子，中间层维度会变为 encoder_dim * expansion_factor
        dropout_p: float = 0.1             # dropout 概率
    ) -> None:
        super(FeedForwardModule, self).__init__()

        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),     # 对输入做层归一化，提升训练稳定性
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),  # 线性层扩展维度
            Swish(),                       # Swish 激活函数，非线性变换
            nn.Dropout(p=dropout_p),       # dropout 防止过拟合
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),  # 投影回原始维度
            nn.Dropout(p=dropout_p),       # 再次 dropout
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)     # 顺序执行模块并输出结果




# class PositionalEncoding(nn.Module):
#     """
#     位置编码模块，来自论文 "Attention Is All You Need"。
#     PE(pos, 2i)   = sin(pos / (10000^(2i / d_model)))
#     PE(pos, 2i+1) = cos(pos / (10000^(2i / d_model)))
#     """

#     def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
#         super(PositionalEncoding, self).__init__()
#         # 初始化一个大小为 [max_len, d_model] 的位置编码矩阵，禁止梯度更新
#         pe = torch.zeros(max_len, d_model, requires_grad=False)
#         # 位置序列 [0, 1, ..., max_len-1]，形状为 [max_len, 1]
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         # 指数因子（不同频率），形状为 [d_model/2]
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
#         # 偶数维度使用正弦函数
#         pe[:, 0::2] = torch.sin(position * div_term)
#         # 奇数维度使用余弦函数
#         pe[:, 1::2] = torch.cos(position * div_term)
#         # 增加 batch 维度，变为 [1, max_len, d_model]
#         pe = pe.unsqueeze(0)
#         # 注册为 buffer，随着模型移动device
#         self.register_buffer('pe', pe)

#     def forward(self, length: int) -> Tensor:
#         return self.pe[:, :length]

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
        self.sqrt_dim = math.sqrt(d_model)  # 用于缩放注意力分数

        # QKV 和 位置嵌入线性变换
        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)  # 位置编码不使用偏置

        # dropout 层
        self.dropout = nn.Dropout(p=dropout_p)

        # 可学习的偏置 u 和 v，用于相对位置注意力中的内容偏置和位置偏置
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))

        # 使用 Xavier 初始化偏置项
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        # 输出线性层
        self.out_proj = Linear(d_model, d_model)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
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

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        # 在最后一维前加一列0，用于偏移
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
        # reshape：将 time1 调整到最后维，再移除第一行
        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]
        return pos_score




class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        # 相对位置编码模块
        # self.positional_encoding = PositionalEncoding(d_model)
        self.positional_encoding = RelPositionalEncoding(d_model, max_pos_len=65)
        # LayerNorm：用于预归一化
        self.layer_norm = nn.LayerNorm(d_model)
        # 相对位置多头注意力模块
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        # Dropout 层，用于正则化
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        inputs = inputs.transpose(1, 0)
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
            mask=mask
        )
        # 最后加 dropout
        outputs = self.dropout(outputs)
        return outputs.transpose(1, 0)



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




class ConformerConvModule(nn.Module):
    """
    Conformer 卷积模块：用于在 Transformer 架构中引入局部建模能力。
    """
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConformerConvModule, self).__init__()
        # 保证 kernel_size 为奇数，从而实现对称 padding（same padding）
        assert (kernel_size - 1) % 2 == 0, "kernel_size 必须是奇数"
        assert expansion_factor == 2, "目前仅支持 expansion_factor = 2"

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),  # 归一化，提升训练稳定性
            Transpose(shape=(1, 2)),    # 转为 (B, C, T) 以适配 Conv1d
            PointwiseConv1d(in_channels, in_channels * expansion_factor),  # 通道扩展
            GLU(dim=1),  # 门控线性单元，只保留一半通道，起作用类似残差门控
            DepthwiseConv1d(in_channels, in_channels, kernel_size, padding=(kernel_size - 1) // 2),  # 深度卷积提取局部特征
            nn.BatchNorm1d(in_channels),  # 批归一化，加速收敛
            Swish(),  # 激活函数
            PointwiseConv1d(in_channels, in_channels),  # 通道还原
            nn.Dropout(p=dropout_p),  # Dropout 防止过拟合
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # 输出维度顺序变回 (B, T, C)
        return self.sequential(inputs).transpose(1, 2)




class ConformerBlock(nn.Module):
    """
    Conformer 模块由两个前馈模块夹住多头自注意力模块和卷积模块组成。
    
    参数:
        encoder_dim (int): 输入与输出的维度（即模型维度）
        num_attention_heads (int): 多头注意力的头数
        feed_forward_expansion_factor (int): 前馈模块的扩展因子，输出维度为 encoder_dim * expansion_factor
        conv_expansion_factor (int): 卷积模块的扩展因子（目前只支持2）
        feed_forward_dropout_p (float): 前馈模块的 dropout 概率
        attention_dropout_p (float): 注意力模块的 dropout 概率
        conv_dropout_p (float): 卷积模块的 dropout 概率
        conv_kernel_size (int): 卷积核的大小，建议为奇数以保持输出长度不变
        half_step_residual (bool): 是否使用“半步残差”，为 True 时每个前馈模块残差乘以0.5
    """
    def __init__(
            self,
            encoder_dim: int = 512,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ):
        super(ConformerBlock, self).__init__()

        # 根据是否使用半步残差设置残差因子
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        # 构建 conformer block 的子模块序列
        self.sequential = nn.Sequential(
            # 第一个前馈模块（残差连接）
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            # 多头自注意力模块（残差连接）
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            ),
            # 卷积模块（残差连接）
            ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            ),
            # 第二个前馈模块（残差连接）
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            # 最后添加 LayerNorm 保证归一化
            nn.LayerNorm(encoder_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)




class Conformer(nn.Module):
    def __init__(self, d_model=512, n_spks=600, dropout=0.1):
        super().__init__()
        # 将输入的 40 维梅尔频谱特征映射到 d_model 维（默认 512）
        self.prenet = nn.Linear(40, d_model)

        # 使用 ConformerBlock 作为编码器（原本为 Transformer）
        # ConformerBlock 结合了前馈层、自注意力和卷积模块，增强了时间建模能力
        self.encoder_layer = ConformerBlock(encoder_dim=d_model, conv_dropout_p=0.1)

        # 分类层
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),  # 扩展维度
            nn.Sigmoid(),                     # 使用 Sigmoid 激活函数
            nn.Dropout(dropout),              # dropout 防止过拟合
            nn.Linear(2 * d_model, n_spks),   # 映射到说话人类别数
        )

    def forward(self, mels):
        # 将输入从 40 维投影到 d_model 维，形状变为 (batch_size, time_length, d_model)
        out = self.prenet(mels)
        # 将输入转置为 (time_length, batch_size, d_model)，适配 ConformerBlock 的输入格式
        out = out.permute(1, 0, 2)
        # 通过 ConformerBlock 进行编码
        out = self.encoder_layer(out)
        # 转置回来为 (batch_size, time_length, d_model)
        out = out.transpose(0, 1)
        # 对时间维做平均池化，得到 (batch_size, d_model)
        stats = out.mean(dim=1)
        # 通过分类层得到最终输出 (batch_size, n_spks)
        out = self.pred_layer(stats)
        return out


if __name__ == '__main__':
    model = Conformer()
    x = torch.rand(3, 128, 40)
    y = model(x)
    print(y.shape)