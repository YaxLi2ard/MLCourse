from config import *

class EMA:
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化 shadow 参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """将 shadow 参数加载进 model 中，做 eval 时用"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """恢复原模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

class MetricComputer:
    def __init__(self):
        # 分别记录gen和dis的loss总和及计数
        self.loss_sum_gen = 0
        self.loss_num_gen = 0

        self.loss_sum_dis = 0
        self.loss_num_dis = 0

    # 判别器损失
    @staticmethod
    def d_loss(real_logits, fake_logits):
        loss = F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()
        return loss

    # 判别器损失（R1 regularization 每16步加一次）
    @staticmethod
    def r1_reg(real_img, D):
        real_img.requires_grad = True
        real_logits = D(real_img, None)
        # with autocast():
        # real_logits = D(real_img)
        grad = torch.autograd.grad(outputs=real_logits.sum(), inputs=real_img, create_graph=True)[0]
        return grad.pow(2).reshape(grad.size(0), -1).sum(1).mean()

    # 生成器损失
    @staticmethod
    def g_loss(fake_logits):
        return F.softplus(-fake_logits).mean()

    def update_loss(self, loss, mode):
        if mode == 'gen':
            self.loss_sum_gen += loss
            self.loss_num_gen += 1
        elif mode == 'dis':
            self.loss_sum_dis += loss
            self.loss_num_dis += 1

    def get_loss(self, mode):
        if mode == 'gen':
            return self.loss_sum_gen / self.loss_num_gen
        elif mode == 'dis':
            return self.loss_sum_dis / self.loss_num_dis

    def reset(self):
        self.loss_sum_gen = 0
        self.loss_num_gen = 0

        self.loss_sum_dis = 0
        self.loss_num_dis = 0

def main():
    test_latent = torch.randn([9, gen_zdim]).to(device)
    metric_cpt = MetricComputer()
    ema = EMA(gen, decay=ema_decay)  # 初始化EMA
    start_time = time.time()
    for step in range(90000):
        # 取一批真实图像
        _, real_img = next(enumerate(dataloader))
        real_img = real_img.to(device)
        # 生成随机变量
        latent = torch.randn([batch_size, gen_zdim]).to(device)
        # 生成器生成图像
        # with autocast():
        fake_img = gen(latent, None)
        # 判别器判别图像
        real_logits = dis(real_img, None)
        fake_logits = dis(fake_img.detach(), None)
        # with autocast():
        # real_logits = dis(real_img)
        # fake_logits = dis(fake_img.detach())

        # 判别器损失
        d_loss = metric_cpt.d_loss(real_logits, fake_logits)
        if step % 16 == 0:
            d_loss = d_loss + metric_cpt.r1_reg(real_img, dis)
        # 更新判别器
        opt_dis.zero_grad()
        d_loss.backward()
        opt_dis.step()
        # scaler.scale(d_loss).backward()
        # scaler.step(opt_dis)
        # scaler.update()

        # 生成器损失
        # fake_img = gen(latent, None)
        fake_logits = dis(fake_img, None)
        # with autocast():
        # fake_logits = dis(fake_img)
        g_loss = metric_cpt.g_loss(fake_logits)
        # 更新生成器
        opt_gen.zero_grad()
        g_loss.backward()
        opt_gen.step()

        metric_cpt.update_loss(d_loss.item(), 'dis')
        metric_cpt.update_loss(g_loss.item(), 'gen')

        scheduler_gen.step()
        scheduler_dis.step()

        if use_ema:
            if (step + 1) % 1 == 0:
                ema.update()

        if (step + 1) % 10 == 0:
            end_time = time.time()
            expensive = end_time - start_time
            start_time = time.time()
            d_loss = metric_cpt.get_loss('dis')
            g_loss = metric_cpt.get_loss('gen')
            metric_cpt.reset()
            print(f"[STEP {step+1}] d_loss:{(d_loss):.3f} g_loss:{(g_loss):.3f} lr:{opt_gen.param_groups[0]['lr']:.7f} r:{(real_logits[0].item()):.3f} f:{(fake_logits[0].item()):.3f} time:{expensive:.2f}")
            writer.add_scalar("d_loss", d_loss, step + 1)
            writer.add_scalar("g_loss", g_loss, step + 1)

        if (step + 1) % 30 == 0:
            test_latent = torch.randn([9, gen_zdim]).to(device)
            gen.eval()
            img = gen(test_latent, None)  # [9, 3, 64, 64]
            gen.train()
            grid = vutils.make_grid(img, nrow=3, normalize=True, scale_each=True)
            writer.add_image("generate_img", grid, global_step=step+1)
            if use_ema:
                ema.apply_shadow()
                gen.eval()
                img = gen(test_latent, None)  # [9, 3, 64, 64]
                gen.train()
                ema.restore()
                grid = vutils.make_grid(img, nrow=3, normalize=True, scale_each=True)
                writer.add_image("generate_img_ema", grid, global_step=step+1)

        if (step + 1) % 500 == 0:
            if use_ema:
                ema.apply_shadow()
                torch.save(gen.state_dict(), f'cpt/gen_ema.pt')
                ema.restore()
            torch.save(gen.state_dict(), f'cpt/gen.pt')
            torch.save(dis.state_dict(), f'cpt/dis.pt')

# python -m train
if __name__ == '__main__':
    main()