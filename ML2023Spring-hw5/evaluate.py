import torch
import sacrebleu
from tqdm import tqdm

def batch_forward_evaluate(model, src_batch, max_len=100, sos_id=2, eos_id=3, pad_id=0):
    """
    对一个 batch 的源句子进行 greedy 解码，返回预测 id 序列列表
    输入:
        src_batch: [B, src_len]
    输出:
        List[List[int]]，每个样本预测出的 token id 列表（不含 <sos>）
    """
    device = next(model.parameters()).device
    B = src_batch.size(0)

    # 1. 构造 src mask
    src_mask = src_batch.eq(pad_id)

    # 2. 编码器只调用一次，得到 encoder_hidden_states
    encoder_outputs = model.encoder(src_batch, attention_mask=~src_mask)
    # encoder_outputs.last_hidden_state shape: [B, src_len, d_model]

    # 3. 初始化 decoder 输入为 <sos>
    ys = torch.full((B, 1), sos_id, dtype=torch.long, device=device)  # [B, 1]

    # 4. 缓存past_key_values，初始为None
    past_key_values = None

    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(max_len):
        # decoder输入当前时间步 token，及缓存
        decoder_outputs = model.decoder(
            input_ids=ys[:, -1:],  # 仅传入最后一个 token
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=~src_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        logits = model.output_fc(decoder_outputs.last_hidden_state[:, 0, :])  # [B, vocab_size]
        next_tokens = logits.argmax(dim=-1, keepdim=True)  # [B, 1]

        ys = torch.cat([ys, next_tokens], dim=1)  # 拼接预测结果
        past_key_values = decoder_outputs.past_key_values  # 更新缓存

        # 标记已经生成<eos>的句子
        finished = finished | (next_tokens.squeeze(1) == eos_id)

        if finished.all():
            break

    ys = ys.tolist()
    return ys


def evaluate_bleu(model, dataloader, metric_cpt, tokenizer_src, tokenizer_tgt, device):
    """
    在验证集上计算计算平均 BLEU 分数。
    """
    references = []
    hypotheses = []

    with torch.no_grad():  # 评估不需要梯度
        for (src_batch, tgt_batch) in tqdm(dataloader):
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
            #  batch_forward_evaluate
            pred_ids_list = batch_forward_evaluate(model, src_batch,
                                                   max_len=100,
                                                   sos_id=tokenizer_tgt.bos_id(),
                                                   eos_id=tokenizer_tgt.eos_id(),
                                                   pad_id=tokenizer_src.pad_id())

            for pred_ids, tgt_ids in zip(pred_ids_list, tgt_batch):
                tgt_ids = tgt_ids.tolist()
                if tokenizer_tgt.eos_id() in tgt_ids:
                    tgt_ids = tgt_ids[:tgt_ids.index(tokenizer_tgt.eos_id())]
                if tokenizer_tgt.eos_id() in pred_ids:
                    pred_ids = pred_ids[:pred_ids.index(tokenizer_tgt.eos_id())]
                reference_tokens = tokenizer_tgt.decode_piece(tgt_ids[1:])  # 自定义 decode_tokens()
                hypothesis_tokens = tokenizer_tgt.decode_piece(pred_ids[1:])
                # 将 token list 拼接为字符串（sacrebleu 要求）
                reference_str = " ".join(reference_tokens)
                hypothesis_str = " ".join(hypothesis_tokens)

                references.append(reference_str)
                hypotheses.append(hypothesis_str)

        # sacrebleu 计算 bleu
        bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize='zh')
        return bleu.score


def evaluate_loss(model, dataloader, metric_cpt, device):
    """
    在验证集上计算分类损失
    """
    with torch.no_grad():  # 评估不需要梯度
        for (src_batch, tgt_batch) in tqdm(dataloader):
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
            yp = model(src_batch, tgt_batch)
            metric_cpt.loss_cpt(yp, tgt_batch, mode='val')

        loss = metric_cpt.get_loss('val')
        metric_cpt.reset('val')
        return loss