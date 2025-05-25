import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class FoodClsDataset(Dataset):
    def __init__(self, data_dir, mode='train', tta=0, transform=None, transform2=None):
        """
        :param data_dir: 图像所在文件夹路径
        :param mode: 'train' | 'val' | 'test'
        :param tta: 若为0,则无TTA;否则执行TTA次数
        :param transform: 图像变换函数
        """
        super(FoodClsDataset, self).__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.tta = tta
        self.transform = transform
        self.transform2 = transform2

        self.samples = []
        for filename in sorted(os.listdir(data_dir)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            path = os.path.join(data_dir, filename)
            # 训练/验证集：文件名格式为 label_id.jpg
            if mode in ['train', 'val']:
                name = os.path.splitext(filename)[0]
                label_str = name.split('_')[0]
                label = int(label_str)
            else:
                # 测试集：无标签
                label = -1
            self.samples.append((path, label))
        # print(self.samples[:10])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert('RGB')  # 转换为RGB图像

        if self.tta <= 0:
            img = self.transform(img)  # 返回 [3, 256, 256]
            return img, label
        else:
            imgs = []
            imgs.append(self.transform(img))
            for _ in range(self.tta - 1):
                imgs.append(self.transform2(img))
            # 返回 [tta, 3, 256, 256]
            return torch.stack(imgs, dim=0), label




if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = FoodClsDataset('/home/aistudio/data/data339209/train', mode='train', tta=0, transform=train_transform)
    print(len(dataset))
    x, y = dataset[0]
    print(x.shape, y)