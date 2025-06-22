from config import *

def main():
    max_acc = 0
    print('Training...')
    for epoch in range(epochs):
        print(f"[Epoch {epoch + 1}] lr:{optimizer.param_groups[0]['lr']:.7f}")
        for batch_idx, data in enumerate(dataloader_train):
            x, y = data
            x, y = x.to(device), y.to(device)
            # with autocast():
            #     yp = model(x)
            yp = model(x)
            # 计算损失
            loss = metric_cpt.loss_cpt(yp, y, mode='train')
            acc = metric_cpt.acc_cpt(yp, y, mode='train')
            # 反向传播和优化
            optimizer.zero_grad()  # PyTorch 清空梯度
            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            if (batch_idx + 1) % 100 == 0:
                print(f'[Train {batch_idx + 1}/{len(dataloader_train)}] Loss:{loss.item():.5f} Acc:{acc:.3f}')

        # 每个 epoch 统计一次 loss 和 acc
        loss = metric_cpt.get_loss('train')
        acc = metric_cpt.get_acc('train')
        metric_cpt.reset('train')
        print("\033[0;31;40m" +
              f'[Train {epoch + 1}] Loss:{loss:.5f} Acc:{acc:.3f}' +
              "\033[0m")

        model.eval()  # 进入评估模式
        loss_val, acc_val = evaluate()
        model.train()  # 切换回训练模式
        print("\033[0;31;40m" +
              f'[Evaluate] Loss:{loss_val:.5f} Acc:{acc_val:.3f}' +
              "\033[0m")

        writer.add_scalars("Loss", {
            "train": loss,
            "val": loss_val
        }, epoch + 1)
        writer.add_scalars("Accuracy", {
            "train": acc,
            "val": acc_val
        }, epoch + 1)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()  # 进入评估模式
            loss_val, acc_val = evaluate2()
            model.train()  # 切换回训练模式
            print("\033[0;31;40m" +
                  f'[Evaluate] Loss:{loss_val:.5f} Acc:{acc_val:.3f}' +
                  "\033[0m")
        
            if (epoch + 1) > 100 and acc_val >= max_acc:
                max_acc = acc_val
                save_cpt = model.state_dict()
                torch.save(save_cpt, f'cpt/trans_{max_acc:.5f}.pt')


def evaluate():
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader_val):
            x, y = data
            x, y = x.to(device), y.to(device)
            yp = model(x)
            metric_cpt.loss_cpt(yp, y, mode='val')
            metric_cpt.acc_cpt(yp, y, mode='val')

    loss = metric_cpt.get_loss('val')
    acc = metric_cpt.get_acc('val')
    metric_cpt.reset('val')

    return loss, acc

def evaluate2():
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataset_val2):
            x, y = data
            x, y = x.to(device), y.to(device)  # x [t, 40]
            x = x.unsqueeze(dim=0)  # x [1, t, 40]
            yp = model(x)  # [1, 600]

            all_preds.append(yp.cpu())
            all_labels.append(y.cpu())

        all_labels = torch.tensor(all_labels)  # [b]
        all_preds = torch.cat(all_preds, dim=0)  # [b, 600]
        
        metric_cpt.loss_cpt(all_preds, all_labels, mode='val')
        metric_cpt.acc_cpt(all_preds, all_labels, mode='val')

    loss = metric_cpt.get_loss('val')
    acc = metric_cpt.get_acc('val')
    metric_cpt.reset('val')

    return loss, acc


if __name__ == '__main__':
    main()