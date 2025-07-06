import torch
from model import load_model
from tqdm import tqdm
import os
from torchvision.utils import save_image
import torchvision.utils as vutils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_img(gen, num=1000, output_dir='output'):
    for i in tqdm(range(num), desc="Generating images"):
        latent = torch.randn([1, 512]).to(device)
        with torch.no_grad():
            img = gen(latent, None)
        save_path = os.path.join(output_dir, f'image_{i:05d}.png')
        save_image(img, save_path)

def generate_grid_img(gen, img_num=21, nrow=7, num=100, output_dir='grid'):
    for i in tqdm(range(num), desc="Generating images"):
        latent = torch.randn([img_num, 512]).to(device)
        with torch.no_grad():
            img = gen(latent, None)
        grid = vutils.make_grid(img, nrow=nrow, normalize=True, scale_each=True)
        save_path = os.path.join(output_dir, f'image_{i:05d}.png')
        save_image(grid, save_path)

def generate_interpolate_image(gen, img_num=3, num=100, steps=7, output_dir='interpolate'):
    for i in tqdm(range(num), desc="Generating images"):
        imgs = []
        for j in range(img_num):
            latent1 = torch.randn([1, 512]).to(device)
            latent2 = torch.randn([1, 512]).to(device)
            alphas = torch.linspace(0, 1, steps).to(device)
            # 插值
            latent_batch = [(1 - alpha) * latent1 + alpha * latent2 for alpha in alphas]
            latent_batch = torch.cat(latent_batch, dim=0)  # 拼成一个 batch，大小为 [steps, 512]
            with torch.no_grad():
                img = gen(latent_batch, None)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
        grid = vutils.make_grid(imgs, nrow=steps, normalize=True, scale_each=True)
        save_path = os.path.join(output_dir, f'image_{i:05d}.png')
        save_image(grid, save_path)
    

if __name__ == '__main__':
    # 加载生成器和权重
    gen = load_model.load_generator(pretrained=False).to(device)
    gen.load_state_dict(torch.load('./cpt/gen_ema.pt', map_location=device), strict=False)
    gen.eval()
    generate_img(gen, num=1, output_dir='output')
    # generate_grid_img(gen, img_num=21, nrow=7, num=10, output_dir='grid')
    # generate_interpolate_image(gen, img_num=1, num=10, steps=7, output_dir='interpolate')