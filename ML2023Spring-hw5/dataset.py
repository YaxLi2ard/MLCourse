import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# class TranslationDataset(Dataset):
#     def __init__(self, src_file, tgt_file, src_tokenizer, tgt_tokenizer, max_len=128):
#         """
#         :param src_file: 源语言文本路径（英文）
#         :param tgt_file: 目标语言文本路径（中文）
#         :param src_tokenizer: 英文分词器（Tokenizer 类）
#         :param tgt_tokenizer: 中文分词器（Tokenizer 类）
#         :param max_len: 最长序列长度
#         """
#         self.src_tokenizer = src_tokenizer
#         self.tgt_tokenizer = tgt_tokenizer
#         self.max_len = max_len
#
#         with open(src_file, 'r', encoding='utf-8') as f:
#             self.src_lines = [line.strip() for line in f if line.strip()]
#         with open(tgt_file, 'r', encoding='utf-8') as f:
#             self.tgt_lines = [line.strip() for line in f if line.strip()]
#
#         assert len(self.src_lines) == len(self.tgt_lines), "源语言和目标语言行数不一致"
#
#     def __len__(self):
#         return len(self.src_lines)
#
#     def __getitem__(self, idx):
#         src_ids = self.src_tokenizer.encode(self.src_lines[idx])[:self.max_len]
#         tgt_ids = self.tgt_tokenizer.encode(self.tgt_lines[idx])[:self.max_len]
#
#         return {
#             'src_input': torch.tensor(src_ids, dtype=torch.long),
#             'tgt_input': torch.tensor(tgt_ids, dtype=torch.long)
#         }

class TranslationDataset(Dataset):
    def __init__(self, src_id_file, tgt_id_file, max_len=1000):
        self.src_ids = []
        self.tgt_ids = []
        # 过滤掉输入长度大于max_len的样本
        with open(src_id_file, 'r', encoding='utf-8') as f_src, \
                open(tgt_id_file, 'r', encoding='utf-8') as f_tgt:
            for src_line, tgt_line in zip(f_src, f_tgt):
                src_ids = list(map(int, src_line.strip().split()))
                tgt_ids = list(map(int, tgt_line.strip().split()))
                if len(src_ids) <= max_len:
                    self.src_ids.append(src_ids)
                    self.tgt_ids.append(tgt_ids)

        assert len(self.src_ids) == len(self.tgt_ids)

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.src_ids[idx], dtype=torch.long),
            torch.tensor(self.tgt_ids[idx], dtype=torch.long)
        )

def collate_fn(batch):
    # 每个batch动态padding
    src_batch = [item[0] for item in batch]
    tgt_batch = [item[1] for item in batch]
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return (src_padded, tgt_padded)

if __name__ == '__main__':
    src_id_file = './DATA/rawdata/processed/train_ids.en'
    tgt_id_file = './DATA/rawdata/processed/train_ids.zh'
    dataset = TranslationDataset(src_id_file, tgt_id_file, max_len=1000)
    x, y = dataset[9]
    print(x)
    print(y)

    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    x, y = next(iter(loader))
    print(x.shape)
    print(y.shape)