import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder
from transformers import BartConfig
import math

# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 用 HuggingFace BartEncoder 和 BartDecoder 替代原来的 nn.Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=2048,
        dropout=0.3,
        max_len=1001,
        pad_id=0
    ):
        super().__init__()
        self.pad_id = pad_id

        # 词嵌入
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)

        # 位置编码
        # self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        # self.pos_decoder = PositionalEncoding(d_model, max_len, dropout)

        # Bart 配置
        config = BartConfig(
            vocab_size=tgt_vocab_size,  # 注意：tgt_vocab_size
            d_model=d_model,
            encoder_layers=num_encoder_layers,
            decoder_layers=num_decoder_layers,
            encoder_attention_heads=nhead,
            decoder_attention_heads=nhead,
            encoder_ffn_dim=dim_feedforward,
            decoder_ffn_dim=dim_feedforward,
            dropout=dropout,
            max_position_embeddings=max_len,
            is_encoder_decoder=True,
        )

        # 编码器和解码器
        self.encoder = BartEncoder(config)
        self.decoder = BartDecoder(config)

        # 输出层
        self.output_fc = nn.Linear(d_model, tgt_vocab_size)

    def make_src_mask(self, src):
        # src: [batch, src_len], bool mask True 表示pad
        return (src == self.pad_id)

    def make_tgt_mask(self, tgt):
        # tgt: [batch, tgt_len]
        tgt_pad_mask = (tgt == self.pad_id)  # padding mask
        tgt_len = tgt.size(1)
        # 生成下三角矩阵 mask
        tgt_sub_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()
        return tgt_pad_mask, tgt_sub_mask

    def forward(self, src, tgt):
        """
        src: [batch, src_len]
        tgt: [batch, tgt_len]
        """
        src_mask = self.make_src_mask(src)  # [batch, src_len]
        tgt_pad_mask, tgt_sub_mask = self.make_tgt_mask(tgt)  # [batch, tgt_len], [tgt_len, tgt_len]

        # 嵌入
        src_emb = self.src_embedding(src)  # [batch, src_len, d_model]
        tgt_emb = self.tgt_embedding(tgt)  # [batch, tgt_len, d_model]

        # Bart的attention_mask是非pad部分为True，需要转一下
        encoder_attention_mask = ~src_mask  # [batch, src_len]
        decoder_attention_mask = ~tgt_pad_mask  # [batch, tgt_len]

        # 编码器前向
        encoder_outputs = self.encoder(
            input_ids=src,
            attention_mask=encoder_attention_mask
        )

        # 解码器前向
        decoder_outputs = self.decoder(
            input_ids=tgt,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_attention_mask,
        )

        # 输出投射
        output = self.output_fc(decoder_outputs.last_hidden_state)  # [batch, tgt_len, tgt_vocab_size]
        return output


if __name__ == '__main__':
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        pad_id=0
    )

    src_batch = torch.tensor([[2, 45, 23, 9, 3, 0, 0]])  # 示例输入
    tgt_batch = torch.tensor([[2, 58, 33, 77, 3, 0, 0]])  # 示例目标

    logits = model(src_batch, tgt_batch)   # 输入不含 <eos>
    print(logits.shape)