from config import *
from evaluate import evaluate_loss, evaluate_bleu

def main():
    # model.eval()  # 进入评估模式
    # loss_val, bleu_score = evaluate(model, dataloader_val, metric_cpt, src_tokenizer, tgt_tokenizer, device)
    # model.train()  # 切换回训练模式
    # print("\033[0;31;40m" +
    #       f'[Evaluate] Loss:{loss_val:.5f} BLEU:{bleu_score:.3f}' +
    #       "\033[0m")
    max_bleu = 0
    print('Training...')
    for epoch in range(epochs):
        print(f"[Epoch {epoch + 1}] lr:{optimizer.param_groups[0]['lr']:.7f}")
        for batch_idx, data in enumerate(dataloader_train):
            x, y = data
            x, y = x.to(device), y.to(device)
            with autocast():
                yp = model(x, y)
            # 计算损失
            loss = metric_cpt.loss_cpt(yp, y, mode='train')
            # 反向传播和优化
            optimizer.zero_grad()  # PyTorch 清空梯度
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (batch_idx + 1) % 500 == 0:
                print(f'[Train {batch_idx + 1}/{len(dataloader_train)}] Loss:{loss.item():.5f}')
                # yp = torch.argmax(yp, dim=2)
                # print(yp[0])
                
        scheduler.step()
        # 每个 epoch 统计一次 loss 和 acc
        loss = metric_cpt.get_loss('train')
        metric_cpt.reset('train')
        print("\033[0;31;40m" +
              f'[Train {epoch + 1}] Loss:{loss:.5f}' +
              "\033[0m")

        if (epoch + 1) % 3 == 0:
            model.eval()  # 进入评估模式
            loss_val = evaluate_loss(model, dataloader_val, metric_cpt, device)
            bleu_score = evaluate_bleu(model, dataloader_val, metric_cpt, src_tokenizer, tgt_tokenizer, device)
            model.train()  # 切换回训练模式
            print("\033[0;31;40m" +
                  f'[Evaluate] Loss:{loss_val:.5f} BLEU:{bleu_score:.3f}' +
                  "\033[0m")

            writer.add_scalars("Loss", {
                "train": loss,
                "val": loss_val
            }, epoch + 1)
            writer.add_scalar("BLEU", bleu_score, epoch + 1)

            if bleu_score >= max_bleu:
                max_bleu = bleu_score
                save_cpt = model.state_dict()
                if (epoch + 1) >= 0:
                    torch.save(save_cpt, f'cpt/transformer_{max_bleu:.3f}.pt')
            
        elif (epoch + 1) % 1 == 0:
            model.eval()  # 进入评估模式
            loss_val = evaluate_loss(model, dataloader_val, metric_cpt, device)
            model.train()  # 切换回训练模式
            print("\033[0;31;40m" +
                  f'[Evaluate] Loss:{loss_val:.5f}' +
                  "\033[0m")

            writer.add_scalars("Loss", {
                "train": loss,
                "val": loss_val
            }, epoch + 1)

if __name__ == '__main__':
    main()