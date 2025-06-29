import os
import random

# 设置文件路径
data_root = '../DATA/rawdata/processed'
src_file = f"{data_root}/train_ids.en"  # 英文原文
tgt_file = f"{data_root}/train_ids.zh"  # 中文翻译

# 输出路径
output_root = '../DATA/rawdata/split'
train_src = f"{output_root}/train_ids.en"
train_tgt = f"{output_root}/train_ids.zh"
dev_src = f"{output_root}/val_ids.en"
dev_tgt = f"{output_root}/val_ids.zh"

# 验证集占比（例如 0.1 表示 10%）
dev_ratio = 0.1
random_seed = 999

# 读取文件内容
with open(src_file, "r", encoding="utf-8") as f:
    src_lines = f.readlines()
with open(tgt_file, "r", encoding="utf-8") as f:
    tgt_lines = f.readlines()

assert len(src_lines) == len(tgt_lines), "英中句子数量不一致！"

# 设置随机种子确保可复现
random.seed(random_seed)

# 打乱顺序并划分
indices = list(range(len(src_lines)))
random.shuffle(indices)

dev_size = int(len(indices) * dev_ratio)
dev_indices = set(indices[:dev_size])
train_indices = set(indices[dev_size:])

# 写入训练集
with open(train_src, "w", encoding="utf-8") as f_src, open(train_tgt, "w", encoding="utf-8") as f_tgt:
    for i in train_indices:
        f_src.write(src_lines[i].strip() + "\n")
        f_tgt.write(tgt_lines[i].strip() + "\n")

# 写入验证集
with open(dev_src, "w", encoding="utf-8") as f_src, open(dev_tgt, "w", encoding="utf-8") as f_tgt:
    for i in dev_indices:
        f_src.write(src_lines[i].strip() + "\n")
        f_tgt.write(tgt_lines[i].strip() + "\n")

print(f"数据划分完成：训练集 {len(train_indices)} 条，验证集 {len(dev_indices)} 条。")
