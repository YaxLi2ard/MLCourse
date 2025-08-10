from setting import *

def train():
    print('Training...')
    model.eval()  # 进入评估模式
    (miou, mdice) = evaluate()
    model.train()  # 切换回训练模式
    print("\033[0;31;40m" +
          f"[Evaluate] miou:{miou:.3f} dice:{mdice:.3f}" +
          "\033[0m")
    for epoch in range(epochs):
        print(f"[Epoch {epoch + 1}] lr:{optimizer.param_groups[0]['lr']:.7f}")
        start_time = end_time = time.time()
        for batch_idx, data in enumerate(dataloader_train):
            x, y = data
            x, y = x.to(device), y.to(device)
            
            y = y.unsqueeze(1).float()  # 先加一个 channel 维度变成 [b, 1, h, w]
            common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
            y_down = F.interpolate(y, scale_factor=1/common_stride, mode='nearest')  # 下采样4倍
            y = y_down.squeeze(1).long()  # 再去掉 channel 维度，回到 [b, h/4, w/4]
            
            with autocast():  # device_type=device.type
                output = model(x)

            # 计算损失
            losses = criterion(output, y) # {'loss_xx':tensor, 'loss_i_xx':tensor}

            loss_ce = 0.0
            loss_dice = 0.0
            loss_mask = 0.0
            for k in list(losses.keys()):
                if k in weight_dict:
                    losses[k] *= criterion.weight_dict[k]  # 乘上这个loss的权重
                    if '_ce' in k:
                        loss_ce += losses[k]
                    elif '_dice' in k:
                        loss_dice += losses[k]
                    elif '_mask' in k:
                        loss_mask += losses[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            loss = loss_ce + loss_dice + loss_mask
            
            # 反向传播和优化
            optimizer.zero_grad()  # PyTorch 清空梯度
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            metric_logger.update('train_loss', loss.item(), num=1)
            
            # scheduler.step()
            
            if (batch_idx + 1) % 10 == 0:
                end_time = time.time()
                print(f'[Train {batch_idx + 1}/{len(dataloader_train)}] Loss:{loss.item():.5f} time:{(end_time-start_time):.2f}s')
                start_time = time.time()
                
        scheduler.step()
        # 每个 epoch 统计一次 loss
        loss = metric_logger.get_metric('train_loss')
        metric_logger.reset('train_loss')
        print("\033[0;31;40m" +
              f"[Train {epoch + 1}] Loss:{loss:.5f}" +
              "\033[0m")
        writer.add_scalars("Loss", {
            "train": loss,
        }, epoch + 1)

        if (epoch + 1) % 5 == 0:
            model.eval()  # 进入评估模式
            (miou, mdice) = evaluate()
            model.train()  # 切换回训练模式
            print("\033[0;31;40m" +
                  f"[Evaluate] miou:{miou:.3f} dice:{mdice:.3f}" +
                  "\033[0m")
            writer.add_scalars("miou", {
                "val": miou,
            }, epoch + 1)
            writer.add_scalars('dice', {
                "val": mdice,
            }, epoch + 1)

        if (epoch + 1) % 1 == 0:
            # 可视化分割效果
            model.eval()  # 进入评估模式
            with torch.no_grad():
                x, y = next(iter_val)
                x, y = x.to(device), y.to(device)
                output = model(x)
                mask_img = post_process(output, filter=False, threshold=0.05)  # [b, h, w]
                overlay = get_image_mask_overlay(x[0], mask_img[0], normalize_mean, normalize_std, colormap, alpha=0.5)
                overlay = (overlay * 255).astype(np.uint8)
                # 转成 tensor，并转维度 HWC->CHW
                overlay_tensor = torch.from_numpy(overlay).permute(2, 0, 1)  # [3,256,256]
                # writer.add_image("segmentation", overlay_tensor, global_step=epoch+1)
                overlay0 = get_image_mask_overlay(x[0], y[0].long(), normalize_mean, normalize_std, colormap, alpha=0.5)
                overlay0 = (overlay0 * 255).astype(np.uint8)
                overlay_tensor0 = torch.from_numpy(overlay0).permute(2, 0, 1)  # [3,256,256]
                # 左右拼接：[3, H, W * 2]
                c, h, w = overlay_tensor.shape
                concat_tensor = torch.cat([overlay_tensor, torch.zeros(c, h, 9).byte(), overlay_tensor0], dim=2)
                writer.add_image("segmentation", concat_tensor, global_step=epoch+1)
                
            model.train()  # 切换回训练模式

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'cpt/ttt.pt')

        # if acc_val >= max_acc:
        #     max_acc = acc_val
        #     save_cpt = model.state_dict()
        # if (epoch + 1) >= 50 and (epoch + 1) % 5 == 0:
        #     torch.save(save_cpt, f'cpt/res50+withoutmix_{max_acc:.5f}.pt')

def post_process(output, filter, threshold=0.5):
    mask_cls_results = output["pred_logits"]
    mask_pred_results = output["pred_masks"]
    # 上采样
    mask_pred_results = F.interpolate(
        mask_pred_results,
        scale_factor=cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
        mode="bilinear",
        align_corners=False,
    )
    pred_masks = semantic_inference(mask_cls_results, mask_pred_results)  # [b, num_cls, h, w]
    if filter:
        probs = torch.softmax(pred_masks, dim=1)  # [B, num_cls, H, W]
        # probs = pred_masks
        conf, _ = torch.max(probs, dim=1)  # [B, H, W]
        # print(conf[0,0:10,0:10])
        ignore_mask = conf < threshold  # bool [B, H, W]
        mask_img = torch.argmax(pred_masks, dim=1)  # [B, H, W]
        mask_img[ignore_mask] = 255  # 置信度低的区域变为255
    else:
        mask_img = torch.argmax(pred_masks, dim=1)  # [B, H, W]
    return mask_img
    

def semantic_inference(mask_cls, mask_pred):    
    # mask_cls [b, num_q, num_classed+1] mask_pred [b, num_q, h, w]
    mask_cls = F.softmax(mask_cls, dim=-1)[...,:-1]  # 去掉no-object类
    mask_pred = mask_pred.sigmoid()      
    semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)        
    return semseg

def evaluate():
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader_val, desc="Validating", unit="samples")):
            x, y = data
            x, y = x.to(device), y.to(device)
            output = model(x)

            mask_img = post_process(output, filter=False, threshold=0.5)  # [b, h, w]
            mask_img = mask_img[0]  # [h, w]

            metric_cpt.update(mask_img, y[0])
            
        miou, mdice = metric_cpt.compute()
        metric_cpt.reset()
        return (miou, mdice)

if __name__ == '__main__':
    train()