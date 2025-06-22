import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import os

class DatasetManager:
    def __init__(self, data_dir, valid_ratio=0.1, segment_len=128):
        self.data_dir = data_dir
        self.segment_len = segment_len

        # 读取说话人映射表（speaker -> id）
        with open(Path(data_dir) / "mapping.json") as f:
            mapping = json.load(f)
            self.speaker2id = mapping["speaker2id"]

        # 构建 id -> speaker 列表
        self.id2speaker = [None] * len(self.speaker2id)
        for speaker, idx in self.speaker2id.items():
            self.id2speaker[idx] = speaker

        # 读取训练集 metadata
        with open(Path(data_dir) / "metadata.json") as f:
            metadata = json.load(f)["speakers"]

        # 构造全部训练样本：路径 + 标签
        all_data = []
        for speaker, utterances in metadata.items():
            label = self.speaker2id[speaker]
            for utt in utterances:
                all_data.append((utt["feature_path"], label))

        # 划分训练集和验证集
        random.seed(999)
        random.shuffle(all_data)
        split_idx = int(len(all_data) * (1 - valid_ratio))
        train_data = all_data[:split_idx]
        valid_data = all_data[split_idx:]

        # 解包成路径和标签
        self.train_files, self.train_labels = zip(*train_data) if train_data else ([], [])
        self.valid_files, self.valid_labels = zip(*valid_data) if valid_data else ([], [])

        # 读取测试数据（无标签）
        with open(Path(data_dir) / "testdata.json") as f:
            testdata = json.load(f)["utterances"]
        self.test_files = [utt["feature_path"] for utt in testdata]
        self.test_labels = [-1] * len(self.test_files)

    def get_datasets(self):
        # 返回三个数据集
        train_set = MyDataset(self.data_dir, self.train_files, self.train_labels, self.segment_len, split=True)
        valid_set = MyDataset(self.data_dir, self.valid_files, self.valid_labels, self.segment_len, split=True)
        valid_set_nosplit = MyDataset(self.data_dir, self.valid_files, self.valid_labels, self.segment_len, split=False)
        test_set = MyDataset(self.data_dir, self.test_files, self.test_labels, self.segment_len, split=True)
        test_set_nosplit = MyDataset(self.data_dir, self.test_files, self.test_labels, self.segment_len, split=False)
        return train_set, valid_set, valid_set_nosplit, test_set, test_set_nosplit

    def get_id2speaker(self):
        # 返回 id -> speaker 的列表
        return self.id2speaker

class MyDataset(Dataset):
    def __init__(self, data_dir, file_list, label_list, segment_len=128, split=True):
        self.data_dir = data_dir            # 数据目录路径
        self.file_list = file_list          # 特征文件路径列表
        self.label_list = label_list        # 标签列表（整数）
        self.segment_len = segment_len      # 片段长度
        self.split = split                  # 是否对特征进行分割/填充处理

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        feat_path = self.file_list[index]       # 获取该样本的特征路径
        label = self.label_list[index]          # 获取该样本的标签

        # 加载 mel-spectrogram 特征（形状如 [帧数, 维度]）
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        if self.split:
            if len(mel) > self.segment_len:
                # 如果帧数大于 segment_len，则随机裁剪一段长度为 segment_len 的序列
                start = random.randint(0, len(mel) - self.segment_len)
                mel = mel[start:start + self.segment_len]
            else:
                # 如果帧数不足，进行零填充
                pad_len = self.segment_len - len(mel)
                mel = torch.nn.functional.pad(mel, (0, 0, 0, pad_len))  # pad 到 [segment_len, 维度]

        mel = torch.FloatTensor(mel)
        label = torch.tensor(label, dtype=torch.long)

        return mel, label


if __name__ == '__main__':
    data_dir = '/root/autodl-tmp/MLCourseDataset/ml2023spring-hw4/Dataset'
    dataset_manager = DatasetManager(data_dir, valid_ratio=0.1, segment_len=128)
    train_set, valid_set, test_set = dataset_manager.get_datasets()
    id2speaker = dataset_manager.get_id2speaker()
    print(len(train_set))
    print(len(valid_set))
    print(len(test_set))
    x, y = train_set[9]
    print(x.shape, y)
    print(id2speaker[1])