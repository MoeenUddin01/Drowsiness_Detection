# src/data/dataloader.py

from torchvision import datasets
from torch.utils.data import DataLoader
from src.data.transforms import train_transform, test_transform

def get_datasets(train_dir='datas/processed/train', test_dir='datas/processed/test'):
    """
    Load training and testing datasets using ImageFolder and specified transforms.
    """
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    return train_dataset, test_dataset


def get_dataloaders(batch_size=32, num_workers=4, shuffle=True,
                    train_dir='datas/processed/train', test_dir='datas/processed/test'):
    """
    Create DataLoader objects for training and testing datasets.
    """
    train_dataset, test_dataset = get_datasets(train_dir=train_dir, test_dir=test_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(batch_size=16)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")
