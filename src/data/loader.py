# src/data/dataloader.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from collections import Counter
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


def calculate_class_weights(dataset):
    """
    Calculate class weights to handle class imbalance.
    Uses inverse frequency weighting: weight = total_samples / (num_classes * class_samples)
    
    Args:
        dataset: PyTorch dataset with labels
        
    Returns:
        torch.Tensor: Class weights tensor
    """
    # Count samples per class
    class_counts = Counter()
    for _, label in dataset:
        class_counts[label] += 1
    
    total_samples = len(dataset)
    num_classes = len(class_counts)
    
    # Calculate weights (inverse frequency)
    weights = []
    for class_idx in sorted(class_counts.keys()):
        class_count = class_counts[class_idx]
        # Inverse frequency weighting
        weight = total_samples / (num_classes * class_count)
        weights.append(weight)
        print(f"  Class {class_idx} ({dataset.classes[class_idx]}): {class_count} samples, weight: {weight:.4f}")
    
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(batch_size=16, num_workers=1, shuffle=True, calculate_weights=True):
    """
    Create DataLoader objects for training and testing datasets.
    
    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        shuffle (bool): Whether to shuffle training data.
        calculate_weights (bool): Whether to calculate class weights for imbalanced datasets.
        
    Returns:
        train_loader, test_loader: PyTorch DataLoader objects
        train_dataset.classes: List of class names
        class_weights: torch.Tensor of class weights (or None if calculate_weights=False)
    """
    train_dataset, test_dataset = get_datasets()
    
    # Auto-detect number of classes
    num_classes = len(train_dataset.classes)
    print(f"\n✅ Auto-detected {num_classes} classes: {train_dataset.classes}")
    print(f"📊 Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Calculate class weights if requested
    class_weights = None
    if calculate_weights:
        print("\n📈 Calculating class weights for balanced training:")
        class_weights = calculate_class_weights(train_dataset)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader, train_dataset.classes, class_weights


if __name__ == "__main__":
    # Quick test to verify DataLoader
    train_loader, test_loader, class_names, class_weights = get_dataloaders(calculate_weights=False)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")
    print(f"Classes: {class_names}")