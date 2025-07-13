# from https://github.com/rosinality/glow-pytorch
import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x))  # 取绝对值后再取对数，常用于稳定性更好的 log|scale|

class ActNorm(nn.Module):
    """
    Activation Normalization，激活归一化模块。
    类似 BatchNorm，但它是对每个通道设置 learnable 的 shift（loc）和 scale 参数。
    在第一次 forward 时使用输入数据初始化 loc 和 scale（初始化只执行一次）。
    """
    def __init__(self, in_channel, logdet=True):
        super().__init__()
        # 初始化可学习参数 loc（偏移）和 scale（缩放）
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        # 用于标记是否完成初始化
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        # 使用第一批输入数据初始化 loc 和 scale，使输出均值为0，方差为1。
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = flatten.mean(1).view(1, -1, 1, 1)
            std = flatten.std(1).view(1, -1, 1, 1)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)
        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet
        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        # 反向变换，用于生成过程。
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    """
    可逆的 1x1 卷积（全通道线性变换），输入输出 shape 不变。
    """
    def __init__(self, in_channel):
        super().__init__()
        weight = torch.randn(in_channel, in_channel)
        # q, _ = torch.qr(weight)  # 初始化为正交矩阵
        q, _ = torch.linalg.qr(weight, mode="reduced")  # 初始化为正交矩阵
        self.weight = nn.Parameter(q.unsqueeze(2).unsqueeze(3))  # 变成卷积权重格式

    def forward(self, input):
        _, _, height, width = input.shape
        out = F.conv2d(input, self.weight)
        logdet = height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        return out, logdet

    def reverse(self, output):
        # 使用逆矩阵做卷积以恢复输入
        return F.conv2d(output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class InvConv2dLU(nn.Module):
    """
    LU 分解形式的可逆 1x1 卷积，减少参数数量并提高数值稳定性。
    """
    def __init__(self, in_channel):
        super().__init__()
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)  # 对角元素
        w_u = np.triu(w_u, 1)  # 上三角不含对角线
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        self.register_buffer("w_p", torch.from_numpy(w_p))
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(torch.from_numpy(w_s.copy())))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(torch.from_numpy(w_l))
        self.w_s = nn.Parameter(logabs(torch.from_numpy(w_s.copy())))  # 可学习的 log(|s|)
        self.w_u = nn.Parameter(torch.from_numpy(w_u))

    def calc_weight(self):
        # 恢复 LU 分解矩阵
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)

    def forward(self, input):
        _, _, height, width = input.shape
        weight = self.calc_weight()
        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)
        return out, logdet

    def reverse(self, output):
        weight = self.calc_weight()
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    """
    初始化为零的卷积层（用于生成均值和方差的 prior 网络），可学习。
    """
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))  # 控制输出幅度

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)  # 输出值扩大
        return out


class AffineCoupling(nn.Module):
    """
    仿射耦合层（Affine Coupling Layer），输入一半变换另一半。
    """
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()
        self.affine = affine
        # 构建子网络
        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if affine else in_channel // 2),
        )

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            s = torch.sigmoid(log_s + 2)  # scale 范围限定在 (0,1)
            out_b = (in_b + t) * s
            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)
        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None
        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            s = torch.sigmoid(log_s + 2)
            in_b = out_b / s - t
        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out
        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()
        # 初始化 ActNorm，作用：做激活归一化，让每个通道的输出均值为0，方差为1
        self.actnorm = ActNorm(in_channel)
        # 初始化可逆1x1卷积，可以选择标准的InvConv2d或LU分解版InvConv2dLU
        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)
        # 初始化仿射耦合层（Affine Coupling Layer），通过一半输入计算另一半的变化
        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        """
        正向过程：依次经过 ActNorm -> 可逆卷积 -> 耦合层
        返回输出特征图 和 logdet（变换的 log-Jacobian 行列式，用于求概率）
        """
        out, logdet = self.actnorm(input)  # 激活归一化
        out, det1 = self.invconv(out)      # 可逆卷积
        out, det2 = self.coupling(out)     # 仿射耦合
        # 合并三个步骤的 logdet 值
        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        return out, logdet

    def reverse(self, output):
        """
        反向过程（生成阶段）：按顺序逆转每个变换
        """
        input = self.coupling.reverse(output)     # 耦合层反向（恢复in_b）
        input = self.invconv.reverse(input)       # 卷积反向（乘以权重逆矩阵）
        input = self.actnorm.reverse(input)       # 归一化反向（去 scale/shift）
        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()

        # 每次 squeeze 操作后，通道数会变为原来的4倍（2x2块）
        squeeze_dim = in_channel * 4

        # 构建多个 Flow 模块（堆叠 n_flow 个）
        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split  # 是否在最后将输出一分为二（多尺度建模）

        # 构建生成用的先验网络，用 ZeroConv2d 输出 mean 和 log_std
        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)
        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input):
        """
        正向传播流程：
        1. Squeeze（空间压缩） -> Flow堆叠 -> optional split + 先验计算
        2. 返回输出特征、logdet、log likelihood、latent变量 z
        """
        b_size, n_channel, height, width = input.shape

        # squeeze：每个 2x2 patch 拉成一个通道，H/2 x W/2，C * 4
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0  # 记录所有 flow 模块的 logdet

        # 依次经过 flow 模块
        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        # split 模式下将输出切分为两个通道：一个用于继续 flow，另一个用于采样
        if self.split:
            out, z_new = out.chunk(2, 1)  # 一半走下去，一半作为latent
            mean, log_sd = self.prior(out).chunk(2, 1)  # 用out去预测 latent 的先验分布
            log_p = gaussian_log_p(z_new, mean, log_sd)  # 计算对数似然
            log_p = log_p.view(b_size, -1).sum(1)  # 求和成每个样本的 log概率
        else:
            zero = torch.zeros_like(out)  # 构造 fake condition
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out  # 整个 out 就是 latent

        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
        """
        反向过程（生成）：给定 latent 变量 z，生成图像
        reconstruct 为 True 时使用传入的 eps（例如重建）
        """
        input = output

        if reconstruct:
            # 重建模式：用已有的 eps 拼接 z
            if self.split:
                input = torch.cat([output, eps], 1)
            else:
                input = eps
        else:
            # 采样模式：使用先验网络采样 eps
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)
            else:
                zero = torch.zeros_like(input)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        # 依次反向 flow
        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        # unsqueeze 恢复图像尺寸，通道变为原来的1/4，空间变为2倍
        b_size, n_channel, height, width = input.shape
        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed


class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        super().__init__()

        # 构建多个 Block，每个 Block 都会将通道数加倍
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        # 最后一个 block 不再 split（输出最深层 z）
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine))

    def forward(self, input):
        """
        编码器（从图像编码为 z）
        返回：
          - log_p_sum：所有 latent 的 log 概率和
          - logdet：所有 flow 的变换量
          - z_outs：所有层的 latent 表示
        """
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        # 依次通过每个 block（包括 squeeze、flow、split）
        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det
            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False):
        """
        生成器（从 z_list 生成图像）
        reconstruct=True 时表示使用已有 z 重建图像
        """
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                # 最后一层没有split，用z自身当作输入
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)
            else:
                # 其他层用上层的输出 + 当前层 z
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

        return input

if __name__ == '__main__':
    model = Glow(
        in_channel=3,
        n_flow=32,
        n_block=4,
        affine=True,
        conv_lu=True
    )
    x = torch.randn(3, 3, 64, 64)
    log_p, logdet, _ = model(x)
    print(log_p.shape)
    print(logdet.shape)
