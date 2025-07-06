from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision
from torchvision import transforms as T

class ImgDataset(Dataset):
    def __init__(self, folder, image_size):
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for p in Path(f'{folder}').glob(f'**/*.jpg')]

        self.transform = T.Compose([
            # T.Resize(image_size),
            # T.RandomRotation(10),  # Random rotation
            T.RandomResizedCrop(size=(image_size, image_size), scale=(0.9, 0.9), ratio=(1.0, 1.0)),
            T.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            T.ColorJitter(brightness=0.19, contrast=0.19),  # Slight color adjustments
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

if __name__ == '__main__':
    dataset = ImgDataset('Y:/Dataset/动漫人脸生成/faces/faces', (64, 64))
    print(dataset[0].shape)