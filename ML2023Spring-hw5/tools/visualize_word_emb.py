import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from tokenize_ import Tokenizer
import os
import sys
sys.path.append('../')
from model.Transformer import Transformer
from itertools import cycle

def get_word_embedding(word, tokenizer, embedding_matrix):
    """
    获取一个词的嵌入表示，若被分成多个子词，则返回它们嵌入的平均。
    """
    token_ids = tokenizer.sp.encode(word, out_type=str)
    token_ids = tokenizer.sp.encode(word, out_type=int)
    if token_ids[0] == '1976':
        token_ids = token_ids[1:]
    embed = embedding_matrix[token_ids]  # [n_subword, d_model]
    return embed.mean(dim=0)  # [d_model]


def visualize_embeddings_zh():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer("ted2020.model")
    model = Transformer(src_vocab_size=tokenizer.vocab_size(),
                        tgt_vocab_size=tokenizer.vocab_size())
    model.load_state_dict(torch.load("../cpt/transformer_25.436.pt", map_location=device), strict=False)
    model.to(device)
    model.eval()
    print("模型加载完成")

    embedding_matrix = model.decoder.embed_tokens.weight.data  # [vocab_size, d_model]

    word_groups = {
        "動物": [
            "貓", "狗", "虎", "兔", "牛", "猴", "龍", "鳥", "魚", "馬",
        ],
        "數字": [
            "七", "八", "九", "個", "十", "百", "千", "萬",
        ],
        "動作": [
            "吃", "走", "跑", "睡", "寫", "讀", "說", "看", "唱", "跳"
        ],
        # "交通工具": [
        #     "汽車", "火車", "飛機", "自行車", "巴士", "船", "地鐵", "摩托車"
        # ],
        "时间": [
            "分", "秒", "時", "天", "日", "周", "月"
        ],
        "身體": [
            "胳膊", "手掌", "頭髮", "脖子", "眼睛", "耳朵", "嘴巴", "肚子"
        ],
        "顏色": [
            "紅", "藍", "綠", "黃", "黑", "橙", "紫", "青", "棕"
        ],
        "水果": [
            "蘋果", "香蕉", "葡萄", "橙子", "西瓜", "菠蘿", "芒果", "草莓",
        ]
    }

    words = []
    embeddings = []
    colors = []

    for group, word_list in word_groups.items():
        for word in word_list:
            vec = get_word_embedding(word, tokenizer, embedding_matrix)
            embeddings.append(vec.cpu().numpy())
            words.append(word)
            colors.append(group)

    embeddings = np.stack(embeddings)
    print("词嵌入完成")

    tsne = TSNE(n_components=2, perplexity=5, init='pca', random_state=39, n_jobs=1)
    # reduced = tsne.fit_transform(embeddings)
    reduced = PCA(n_components=2).fit_transform(embeddings)

    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题
    plt.figure(figsize=(9, 6))

    # 分配颜色
    unique_groups = list(word_groups.keys())
    color_cycle = cycle(plt.get_cmap("tab10").colors)
    palette = {group: next(color_cycle) for group in unique_groups}

    for i, word in enumerate(words):
        plt.scatter(reduced[i, 0], reduced[i, 1], color=palette[colors[i]],
                    label=colors[i] if word == word_groups[colors[i]][0] else "")
        plt.text(reduced[i, 0] + 0.001, reduced[i, 1] + 0.001, word, fontsize=9)

    plt.axis('equal')

    plt.title("词语嵌入可视化")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig("embedding_visualization.png", dpi=300)
    plt.show()
    print("✅ 可视化完成，结果已保存为 embedding_visualization.png")

def visualize_embeddings_en():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer("./tools/ted2020.model")
    model = Transformer(src_vocab_size=tokenizer.vocab_size(),
                        tgt_vocab_size=tokenizer.vocab_size())
    model.load_state_dict(torch.load("./cpt/transformer_25.436.pt", map_location=device), strict=False)
    model.to(device)
    model.eval()
    print("模型加载完成")

    embedding_matrix = model.encoder.embed_tokens.weight.data  # 使用 encoder 的嵌入

    word_groups = {
        "animals": [
            "cat", "dog", "tiger", "rabbit", "cow", "monkey", "dragon", "bird", "fish", "horse",
        ],
        "numbers": [
            "seven", "eight", "nine", "one", "ten", "hundred", "thousand", "million",
        ],
        "actions": [
            "eat", "walk", "run", "sleep", "write", "read", "say", "see", "sing", "jump"
        ],
        "time": [
            "minute", "second", "hour", "day", "week", "month", "year"
        ],
        "body": [
            "arm", "hand", "hair", "neck", "eye", "ear", "mouth", "belly"
        ],
        "colors": [
            "red", "blue", "green", "yellow", "black", "orange", "purple", "cyan", "brown"
        ],
        "fruits": [
            "apple", "banana", "grape", "orange", "watermelon", "pineapple", "mango", "strawberry"
        ]
    }

    words = []
    embeddings = []
    colors = []

    for group, word_list in word_groups.items():
        for word in word_list:
            vec = get_word_embedding(word, tokenizer, embedding_matrix)
            embeddings.append(vec.cpu().numpy())
            words.append(word)
            colors.append(group)

    embeddings = np.stack(embeddings)
    print("詞嵌入完成")

    tsne = TSNE(n_components=2, perplexity=5, init='pca', random_state=39, n_jobs=1)
    reduced = PCA(n_components=2).fit_transform(embeddings)

    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(9, 6))

    unique_groups = list(word_groups.keys())
    color_cycle = cycle(plt.get_cmap("tab10").colors)
    palette = {group: next(color_cycle) for group in unique_groups}

    for i, word in enumerate(words):
        plt.scatter(reduced[i, 0], reduced[i, 1], color=palette[colors[i]],
                    label=colors[i] if word == word_groups[colors[i]][0] else "")
        plt.text(reduced[i, 0] + 0.001, reduced[i, 1] + 0.001, word, fontsize=9)

    plt.axis('equal')
    plt.title("English Word Embedding Visualization")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig("embedding_en_visualization.png", dpi=300)
    plt.show()
    print("✅ 可視化完成，結果已保存為 embedding_en_visualization.png")

def visualize_cn_en_embeddings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer("./tools/ted2020.model")
    model = Transformer(src_vocab_size=tokenizer.vocab_size(),
                        tgt_vocab_size=tokenizer.vocab_size())
    model.load_state_dict(torch.load("./cpt/transformer_25.436.pt", map_location=device), strict=False)
    model.to(device)
    model.eval()
    print("模型加载完成")

    # 取 encoder 和 decoder 的 embedding 矩阵
    encoder_embedding = model.encoder.embed_tokens.weight.data  # encoder embedding，英文词
    decoder_embedding = model.decoder.embed_tokens.weight.data  # decoder embedding，中文词

    # 三个类别的中英文词对
    word_pairs = {
        # "動物": [
        #     ("貓", "cat"),
        #     ("狗", "dog"),
        #     ("虎", "tiger"),
        #     ("牛", "cattle"),
        #     ("魚", "fish"),
        # ],
        "顏色": [
            ("紅", "red"),
            ("藍", "blue"),
            ("綠", "green"),
            ("黃", "yellow"),
            ("黑", "black"),
        ],
        # "動作": [
        #     ("吃", "eat"),
        #     ("走", "walk"),
        #     ("跑", "run"),
        #     ("睡覺", "sleep"),
        #     ("寫", "write"),
        # ],
    }

    words = []
    embeddings = []
    colors = []
    langs = []  # 标记中文 or 英文

    for group, pairs in word_pairs.items():
        for cn_word, en_word in pairs:
            # 中文用 decoder embedding
            cn_vec = get_word_embedding(cn_word, tokenizer, decoder_embedding)
            embeddings.append(cn_vec.cpu().numpy())
            words.append(cn_word)
            colors.append(group)
            langs.append("中文")

            # 英文用 encoder embedding
            en_vec = get_word_embedding(en_word, tokenizer, encoder_embedding)
            embeddings.append(en_vec.cpu().numpy())
            words.append(en_word)
            colors.append(group)
            langs.append("英文")

    embeddings = np.stack(embeddings)
    print("词嵌入完成")

    # 降维，t-SNE 可能慢，这里先用 PCA，也可换成 TSNE
    reduced = PCA(n_components=2).fit_transform(embeddings)

    plt.rcParams['font.family'] = 'SimHei'  # 中文显示
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 7))

    unique_groups = list(word_pairs.keys())
    color_cycle = cycle(plt.get_cmap("tab10").colors)
    palette = {group: next(color_cycle) for group in unique_groups}

    # 画点时区分中文和英文形状：中文用圆圈 'o'，英文用三角 '^'
    marker_map = {"中文": "o", "英文": "^"}

    for i, word in enumerate(words):
        plt.scatter(reduced[i, 0], reduced[i, 1],
                    color=palette[colors[i]],
                    marker=marker_map[langs[i]],
                    s=80,
                    label=f"{colors[i]}-{langs[i]}" if word == word_pairs[colors[i]][0][0] or word == word_pairs[colors[i]][0][1] else "")
        # 文字稍微右上偏移
        plt.text(reduced[i, 0] + 0.001, reduced[i, 1] + 0.001, word, fontsize=9)

    plt.axis('equal')
    plt.title("中英文词语嵌入可视化 (encoder vs decoder)")

    # 处理图例去重
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')

    plt.tight_layout()
    plt.savefig("cn_en_embedding.png", dpi=300)
    plt.show()
    print("✅ 可视化完成，结果已保存为 cn_en_embedding.png")











if __name__ == '__main__':
    visualize_embeddings_zh()
    # visualize_embeddings_en()
    # visualize_cn_en_embeddings()