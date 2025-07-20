from config import *

def train():
    print('Training...')
    for epoch in range(epochs):
        print(f"[Epoch {epoch + 1}] lr:{optimizer.param_groups[0]['lr']:.7f}")
        for batch_idx, data in enumerate(dataloader_train):
            x, y = data
            x, y = x.to(device), y.to(device)
            with autocast():  # device_type=device.type
                yp = model(x)
                yp = yp.float()
                # 计算损失
            (loss, mpa, miou) = metric_cpt.cpt_update(yp, y, mode='train')
            # 反向传播和优化
            optimizer.zero_grad()  # PyTorch 清空梯度
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (batch_idx + 1) % 10 == 0:
                print(f'[Train {batch_idx + 1}/{len(dataloader_train)}] Loss:{loss.item():.5f} mpa:{mpa:.3f} miou:{miou:.3f}')

        # 每个 epoch 统计一次 loss 和 acc
        metric = metric_cpt.get_metric('train')
        metric_cpt.reset('train')
        print("\033[0;31;40m" +
              f"[Train {epoch + 1}] Loss:{metric['loss']:.5f} mpa:{metric['mpa']:.3f} miou:{metric['miou']:.3f}" +
              "\033[0m")
        writer.add_scalars("Loss", {
            "train": metric['loss'],
        }, epoch + 1)
        writer.add_scalars("mpa", {
            "train": metric['mpa'],
        }, epoch + 1)
        writer.add_scalars("miou", {
            "train": metric['miou'],
        }, epoch + 1)

        if (epoch + 1) % 5 == 0:
            model.eval()  # 进入评估模式
            metric0 = evaluate()
            model.train()  # 切换回训练模式
            print("\033[0;31;40m" +
                  f"[Evaluate] Loss:{metric0['loss']:.5f} mpa:{metric0['mpa']:.3f} miou:{metric0['miou']:.3f}" +
                  "\033[0m")
            writer.add_scalars("Loss", {
                "val": metric0['loss'],
            }, epoch + 1)
            writer.add_scalars("mpa", {
                "val": metric0['mpa'],
            }, epoch + 1)
            writer.add_scalars("miou", {
                "val": metric0['miou'],
            }, epoch + 1)

        if (epoch + 1) % 1 == 0:
            # 可视化分割效果
            model.eval()  # 进入评估模式
            with torch.no_grad():
                x, y = next(iter_val)
                x, y = x.to(device), y.to(device)
                yp = model(x)
                yp = yp.argmax(dim=1).unsqueeze(dim=1)  # [B, 1, H, W]
                overlay = get_image_mask_overlay(x[0], yp[0], alpha=0.5)
                overlay = (overlay * 255).astype(np.uint8)
                # 转成 tensor，并转维度 HWC->CHW
                overlay_tensor = torch.from_numpy(overlay).permute(2, 0, 1)  # [3,256,256]
                # writer.add_image("segmentation", overlay_tensor, global_step=epoch+1)
                y = F.interpolate(y.float(), size=yp.shape[-2:], mode='nearest')
                overlay0 = get_image_mask_overlay(x[0], y[0].long(), alpha=0.5)
                overlay0 = (overlay0 * 255).astype(np.uint8)
                overlay_tensor0 = torch.from_numpy(overlay0).permute(2, 0, 1)  # [3,256,256]
                # 左右拼接：[3, H, W * 2]
                c, h, w = overlay_tensor.shape
                concat_tensor = torch.cat([overlay_tensor, torch.zeros(c, h, 9).byte(), overlay_tensor0], dim=2)
                writer.add_image("segmentation", concat_tensor, global_step=epoch+1)
                
            model.train()  # 切换回训练模式

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'cpt/ttt.pt')
            
            
        scheduler.step()

        # if acc_val >= max_acc:
        #     max_acc = acc_val
        #     save_cpt = model.state_dict()
        # if (epoch + 1) >= 50 and (epoch + 1) % 5 == 0:
        #     torch.save(save_cpt, f'cpt/res50+withoutmix_{max_acc:.5f}.pt')


def evaluate():
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader_val):
            x, y = data
            x, y = x.to(device), y.to(device)
            yp = model(x)
            metric_cpt.cpt_update(yp, y, mode='val')

        metric = metric_cpt.get_metric('val')
        metric_cpt.reset('val')
        return metric

if __name__ == '__main__':
    train()