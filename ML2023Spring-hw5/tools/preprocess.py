import os
import re


# 全角转半角函数
def full2half(s):
    result = ''
    for char in s:
        code = ord(char)
        # 全角空格
        if code == 0x3000:
            code = 0x0020
        # 全角标点、字符（字母、数字、符号等）
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        result += chr(code)
    return result


# 通用预处理函数
def preprocess_line(text):
    text = full2half(text)  # 全角转半角
    text = text.strip()  # 去除首尾空格
    text = re.sub(r'\s+', ' ', text)  # 多个空白字符变为一个空格
    return text


# 对整个文件进行预处理
def preprocess_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, \
            open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if line == "":
                continue  # 跳过空行
            processed = preprocess_line(line)
            if processed:  # 仍然非空才写入
                fout.write(processed + '\n')


# 示例用法
if __name__ == "__main__":
    input_files = ['test.raw.en', 'test.raw.zh', 'train_dev.raw.en', 'train_dev.raw.zh']
    input_dir = "../DATA/rawdata/ted2020/"
    output_files = ['test_clean.en', 'test_clean.zh', 'train_clean.en', 'train_clean.zh']
    output_dir = "../DATA/rawdata/processed/"
    for i in range(len(input_files)):
        input_file = os.path.join(input_dir, input_files[i])
        output_file = os.path.join(output_dir, output_files[i])
        preprocess_file(input_file, output_file)
        print(f"预处理完成：{input_file} → {output_file}")
