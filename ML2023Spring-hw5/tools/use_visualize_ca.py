import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tokenize_ import Tokenizer
sys.path.append('../')
from model.Transformer import Transformer
import numpy as np


def main():
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ------------------ Step 1: 加载模型和 tokenizer ------------------
    # tokenizer
    tokenizer_src = Tokenizer("ted2020.model")
    tokenizer_tgt = Tokenizer("ted2020.model")  # 英文和中文用同一模型
    # 模型
    model = Transformer(src_vocab_size=tokenizer_src.vocab_size(), tgt_vocab_size=tokenizer_tgt.vocab_size())
    model.to(device)

    model.load_state_dict(torch.load("../cpt/transformer_25.436.pt", map_location=device), strict=False)
    model.eval()
    print('模型加载完成...')

    # ------------------ Step 2: 输入文本编码 ------------------
    text = """ After we finished our lunch, we went to the park and played there for more than two hours. """  # He'll do sports tomorrow if the weather's good.
    src_ids = tokenizer_src.encode(text)  # 加上 <bos> 和 <eos>
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    src_mask = (src_tensor == tokenizer_src.pad_id())

    # ------------------ Step 3: 编码器前向 ------------------
    memory = model.encoder(src_tensor, attention_mask=~src_mask)
    print('编码器前向完成...')

    # ------------------ Step 4: 解码器逐步生成并记录注意力 ------------------
    ys = torch.tensor([[tokenizer_tgt.bos_id()]], dtype=torch.long).to(device)
    output_tokens = []
    attention_maps = []

    for _ in range(101):
        tgt_mask = torch.triu(torch.ones((ys.size(1), ys.size(1)), device=device), diagonal=1).bool()

        # forward with output attention
        out = model.decoder(
            input_ids=ys,
            encoder_hidden_states=memory.last_hidden_state,
            encoder_attention_mask=~src_mask,
            output_attentions=True,  # 获取注意力
            return_dict=True
        )
        # 获取最后一个时间步预测
        logits = model.output_fc(out.last_hidden_state[:, -1])
        next_token = logits.argmax(dim=-1).item()
        output_tokens.append(next_token)
        ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)

        # 保存最后一层交叉注意力（decoder 最后一层）
        # shape: [batch, num_heads, tgt_len, src_len]
        cross_attn = out.cross_attentions[1][0, :, -1, :]  # 取出最后一个位置 [num_heads, src_len]
        attention_maps.append(cross_attn.mean(dim=0).detach().cpu())  # 平均多个头 → [src_len]

        if next_token == tokenizer_tgt.eos_id():
            break

    # ------------------ Step 5: 解码输出中文 ------------------
    output_ids = output_tokens[:-1]  # 去掉 <eos>
    translation = tokenizer_tgt.decode(output_ids)
    print("🌐 英文输入:", text)
    print("🇨🇳 翻译输出:", translation)

    # ------------------ Step 6: 可视化交叉注意力 ------------------
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

    attn_matrix = torch.stack(attention_maps, dim=0)  # [tgt_len, src_len]

    src_tokens = ['<bos>'] + tokenizer_src.sp.encode(text, out_type=str) + ['<eos>']  # 输入英文（列标签）
    tgt_tokens = tokenizer_tgt.sp.encode(translation, out_type=str) + ['<eos>']  # 输出中文（行标签）

    # plt.figure(figsize=(len(tgt_tokens) * 0.6 + 3, len(src_tokens) * 0.3 + 3))
    plt.figure(figsize=(16, 9))
    sns.heatmap(attn_matrix.numpy(),
                xticklabels=src_tokens,
                yticklabels=tgt_tokens,
                cmap='viridis', linewidths=0.5, annot=False, cbar=False)
    plt.xticks(rotation=0)  # 横向显示输入 token
    plt.yticks(rotation=0)  # 行本来也横向显示，这里保留明确指示
    plt.xlabel("输入英文 Token")
    plt.ylabel("生成中文 Token")
    plt.title(f"原文： {text} \n 翻译：{translation}")
    plt.tight_layout()
    plt.savefig("attention_map.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
