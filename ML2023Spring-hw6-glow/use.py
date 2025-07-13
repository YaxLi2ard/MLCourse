import torch
from model.glow import Glow
from tqdm import tqdm
import os
from torchvision.utils import save_image
import torchvision.utils as vutils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_flow = 32
n_block = 4
img_sz = 64
n_channel = 3
temperature = 0.6
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
z_shapes = calc_z_shapes(n_channel, img_sz, n_flow, n_block)

def sampling(batch_sz, z_shapes, temperature=0.7):
    z_sample = []
    for z in z_shapes:
        z_new = torch.randn(batch_sz, *z) * temperature  # temperature控制多样性
        z_sample.append(z_new.to(device))
    return z_sample

def generate_img(model, num=1000, output_dir='output'):
    for i in tqdm(range(num), desc="Generating images"):
        z_sample = sampling(1, z_shapes, temperature=temperature)
        with torch.no_grad():
            img = model.reverse(z_sample).cpu().data
            img = img + 0.5
        save_path = os.path.join(output_dir, f'image_{i:05d}.png')
        save_image(img, save_path)

def generate_grid_img(model, img_num=21, nrow=7, num=100, output_dir='grid'):
    for i in tqdm(range(num), desc="Generating images"):
        z_sample = sampling(img_num, z_shapes, temperature=temperature)
        with torch.no_grad():
            img = model.reverse(z_sample).cpu().data
        grid = vutils.make_grid(img, nrow=nrow, normalize=True, scale_each=True)
        save_path = os.path.join(output_dir, f'image_{i:05d}.png')
        save_image(grid, save_path)

def generate_grid_img2(model, ts=[0.3, 0.5, 0.7], nrow=7, num=100, output_dir='grid'):
    for i in tqdm(range(num), desc="Generating images"):
        imgs = []
        for t in ts:
            z_sample = sampling(nrow, z_shapes, temperature=t)
            with torch.no_grad():
                img = model.reverse(z_sample).cpu().data
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
        grid = vutils.make_grid(imgs, nrow=nrow, normalize=True, scale_each=True)
        save_path = os.path.join(output_dir, f'image_{i:05d}.png')
        save_image(grid, save_path)
    

if __name__ == '__main__':
    # 加载生成器和权重
    model = Glow(
        in_channel=n_channel,
        n_flow=n_flow,
        n_block=n_block,
        affine=True,
        conv_lu=False
    ).to(device)
    checkpoint = torch.load('cpt/cpt_61_20000.pt')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    z_shapes = calc_z_shapes(3, img_sz, n_flow, n_block)
    generate_img(model, num=1, output_dir='output')
    # generate_grid_img(model, img_num=21, nrow=7, num=10, output_dir='grid')
    # generate_grid_img2(model, ts=[0.3, 0.5, 0.7], nrow=7, num=10, output_dir='grid')
    # generate_interpolate_image(gen, img_num=1, num=10, steps=7, output_dir='interpolate')