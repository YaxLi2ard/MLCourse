import math

from config import *

# 计算每一层 latent z 的 shape
def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []
    for i in range(n_block - 1):
        input_size //= 2     # 每个 block 会下采样一半
        n_channel *= 2       # 通道数翻倍
        z_shapes.append((n_channel, input_size, input_size))
    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))  # 最后一层不 split
    return z_shapes

# 计算 loss（负对数似然 NLL）
def calc_loss(log_p, logdet, image_size, n_bins):
    n_pixel = image_size * image_size * 3  # 每张图像的像素总数（3 通道）
    # 损失函数（对数密度变换）
    loss = -math.log(n_bins) * n_pixel          # 对图像做了离散化（bits），加入 log(n_bins)
    loss = loss + logdet + log_p           # 加上模型输出的 logdet 和 log_p
    # 返回三个指标（均值，按像素归一化并以 log2 计）
    return (
        (-loss / (math.log(2) * n_pixel)).mean(),     # 总损失
        (log_p / (math.log(2) * n_pixel)).mean(),     # 负 log-likelihood
        (logdet / (math.log(2) * n_pixel)).mean(),    # log determinant
    )

class MetricLogger:
    def __init__(self):
        # 记录loss总和及计数
        self.loss_sum = 0
        self.loss_num = 0

        self.logp_sum = 0
        self.logp_num = 0

        self.logdet_sum = 0
        self.logdet_num = 0

    def update_loss(self, loss, mode):
        if mode == 'loss':
            self.loss_sum += loss
            self.loss_num += 1
        elif mode == 'log_p':
            self.logp_sum += loss
            self.logp_num += 1
        elif mode == 'logdet':
            self.logdet_sum += loss
            self.logdet_num += 1

    def get_loss(self, mode):
        if mode == 'loss':
            return self.loss_sum / self.loss_num
        elif mode == 'log_p':
            return self.logp_sum / self.logp_num
        elif mode == 'logdet':
            return self.logdet_sum / self.logdet_num

    def reset(self):
        self.__init__()

def train():
    print(-math.log(32) * 64 * 64 * 3)
    # 生成阶段用的 latent z（标准正态 * 温度）
    z_sample = []
    z_shapes = calc_z_shapes(3, img_sz, n_flow, n_block)
    for z in z_shapes:
        z_new = torch.randn(n_sample, *z) * temperature  # temperature控制多样性
        z_sample.append(z_new.to(device))
    # # img = model.reverse(z_sample).cpu().data
    # # print(img.shape)
    # # return 0

    # 损失记录
    metric_logger = MetricLogger()
    print('Start training...')
    start_time = time.time()
    for step in range(iters):
        # 取一批真实图像
        _, img = next(enumerate(dataloader))
        img = img.to(device)
        # 离散化、标准化
        img = img * 255  # 转回[0,255]
        if n_bits < 8:
            img = torch.floor(img / 2 ** (8 - n_bits))  # 离散化
        img = img / n_bins - 0.5  # 标准化到 [-0.5, 0.5]
        # 仅第一次做一次前向推理，不参与梯度更新
        if step == 0:
            with torch.no_grad():
                log_p, logdet, _ = model.module(img + torch.rand_like(img) / n_bins)
            continue
        # 前向传播，加入微小噪声
        log_p, logdet, _ = model(img + torch.rand_like(img) / n_bins)
        logdet = logdet.mean()

        # 计算损失（返回 [总损失 / logp / logdet] ）
        loss, log_p, log_det = calc_loss(log_p, logdet, img_sz, n_bins)

        model.zero_grad()
        loss.backward()
        optim.step()

        scheduler.step()

        metric_logger.update_loss(loss.item(), 'loss')
        metric_logger.update_loss(log_p.item(), 'log_p')
        metric_logger.update_loss(logdet.item(), 'logdet')

        # 打印损失指标
        if (step + 1) % 10 == 0:
            end_time = time.time()
            expensive = end_time - start_time
            start_time = time.time()
            loss = metric_logger.get_loss('loss')
            log_p = metric_logger.get_loss('log_p')
            logdet = metric_logger.get_loss('logdet')
            metric_logger.reset()
            print(f"[STEP {step+1}] loss:{(loss):.3f} log_p:{(log_p):.3f} logdet:{(logdet):.3f} lr:{optim.param_groups[0]['lr']:.7f} time:{expensive:.2f}")
            writer.add_scalar("loss", loss, step + 1)
            writer.add_scalar("log_p", log_p, step + 1)
            writer.add_scalar("logdet", logdet, step + 1)

        # 打印生成图像样本
        if step % 30 == 0:
            with torch.no_grad():
                img = model.reverse(z_sample).cpu().data
            img = img + 0.5
            grid = vutils.make_grid(img, nrow=math.floor(math.sqrt(n_sample)), normalize=True, scale_each=True)
            writer.add_image("generate_img", grid, global_step=step + 1)

        # 保存模型权重
        if step % 500 == 0:
            torch.save(model.state_dict(), f"cpt/model_{str(step + 1).zfill(6)}.pt")
            torch.save(optim.state_dict(), f"cpt/optim_{str(step + 1).zfill(6)}.pt")

if __name__ == '__main__':
    train()