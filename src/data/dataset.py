# src/data/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
from src.data.transforms import get_data_transforms

class DrowsinessDataset(Dataset):
    def __init__(self, data_dir, train=True):
        """
        data_dir:
            datas/processed/train
            datas/processed/test
        """
        self.data_dir = data_dir
        self.transform = get_data_transforms(train)

        self.images = []
        self.labels = []

        class_to_label = {
            "closed_1": 0,
            "opened": 1
        }

        for class_name, label in class_to_label.items():
            class_path = os.path.join(data_dir, class_name)

            if not os.path.exists(class_path):
                continue

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.images.append(img_path)
                self.labels.append(label)

    def __len__(self):
        """Return total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Return one sample"""
        img_path = self.images[index]
        label = self.labels[index]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
