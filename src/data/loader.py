# src/data/dataloader.py

from torch.utils.data import DataLoader
from torchvision import datasets
from src.data.transforms import get_train_transform, get_test_transform  # Updated imports

def get_datasets(train_dir='datas/processed/train', test_dir='datas/processed/test'):
    """
    Load training and testing datasets using ImageFolder and specified transforms.
    
    Args:
        train_dir (str): Path to the training data folder.
        test_dir (str): Path to the testing data folder.
        
    Returns:
        train_dataset, test_dataset: PyTorch ImageFolder datasets
    """
    train_dataset = datasets.ImageFolder(root=train_dir, transform=get_train_transform())
    test_dataset = datasets.ImageFolder(root=test_dir, transform=get_test_transform())
    
    return train_dataset, test_dataset


def get_dataloaders(batch_size=16, num_workers=1, shuffle=True):
    """
    Create DataLoader objects for training and testing datasets.
    
    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        shuffle (bool): Whether to shuffle training data.
        
    Returns:
        train_loader, test_loader: PyTorch DataLoader objects
    """
    train_dataset, test_dataset = get_datasets()
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Quick test to verify DataLoader
    train_loader, test_loader = get_dataloaders()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")
