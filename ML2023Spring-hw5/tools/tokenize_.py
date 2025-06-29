import sentencepiece as spm
import os


def train_sentencepiece(input_files, model_prefix, vocab_size=8000, character_coverage=1.0, model_type='bpe'):
    """
    训练 SentencePiece 分词器
    :param input_files: list[str], 输入文件路径列表（例如中英文 clean 文件）
    :param model_prefix: str, 保存模型和词表的前缀名
    :param vocab_size: int, 词表大小
    :param character_coverage: float, 字符覆盖率（中文建议为 1.0）
    :param model_type: str, 分词模型类型：bpe、unigram、char、word
    """
    combined_file = f"{model_prefix}_combined.txt"

    # 将多个输入文件拼接为一个训练用文件
    with open(combined_file, 'w', encoding='utf-8') as fout:
        for file in input_files:
            with open(file, 'r', encoding='utf-8') as fin:
                for line in fin:
                    fout.write(line.strip() + '\n')

    # 调用 SentencePiece 训练
    spm.SentencePieceTrainer.train(
        input=combined_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols=[],
    )

    print(f"SentencePiece 模型训练完成，模型文件：{model_prefix}.model，词表文件：{model_prefix}.vocab")


class Tokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text):
        # 加上 BOS 和 EOS，返回 token id 列表
        return [self.sp.bos_id()] + self.sp.encode(text, out_type=int) + [self.sp.eos_id()]

    def decode(self, ids):
        return self.sp.decode(ids)

    def decode_piece(self, ids):
        tokens = [self.sp.id_to_piece(i) for i in ids]
        return tokens

    def pad_id(self):
        return self.sp.pad_id()

    def bos_id(self):
        return self.sp.bos_id()

    def eos_id(self):
        return self.sp.eos_id()

    def vocab_size(self):
        return self.sp.get_piece_size()


def encode_file(input_path, output_path, tokenizer):
    """
    将文本文件编码为 ID 序列并保存
    :param input_path: 原始文本文件路径
    :param output_path: 输出 ID 序列文件路径
    :param tokenizer: Tokenizer 对象
    """
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line)
            fout.write(" ".join(map(str, ids)) + "\n")
    print(f"编码完成：{input_path} -> {output_path}")

if __name__ == '__main__':
    tokenizer = Tokenizer("ted2020.model")
    a = tokenizer.sp.encode("我喜欢学习", out_type=str)
    print(a)
    ids = [83, 2458, 7082, 5635, 7331]
    a = tokenizer.sp.decode(ids)
    print(a)
    a = tokenizer.decode_piece(ids)
    print(a)
    # # ---------------------------- 训练分词器 ----------------------------
    # files = ['train_dev.raw.en', 'train_dev.raw.zh']
    # input_dir = "../DATA/rawdata/ted2020/"
    # input_files = [os.path.join(input_dir, file) for file in files]
    # train_sentencepiece(
    #     input_files=input_files,
    #     model_prefix="ted2020",
    #     vocab_size=8000,
    #     character_coverage=1.0,  # 中文一般设为 1.0
    #     model_type="bpe"  # 或 "unigram"
    # )
    # # ---------------------------- 训练分词器 ----------------------------

    # ---------------------------- 编码文件并保存 ----------------------------
    # input_dir = "../DATA/rawdata/processed/"
    # output_dir = "../DATA/rawdata/processed/"

    # src_tokenizer = Tokenizer("ted2020.model")
    # tgt_tokenizer = Tokenizer("ted2020.model")  # 英文和中文用同一模型

    # encode_file(os.path.join(input_dir, 'train_clean.en'), os.path.join(output_dir, 'train_ids.en'), src_tokenizer)
    # encode_file(os.path.join(input_dir, 'train_clean.zh'), os.path.join(output_dir, 'train_ids.zh'), tgt_tokenizer)