# src/data/transforms.py

from torchvision import transforms

def get_train_transform(image_size=(128, 128)):
    """
    Returns a torchvision transform for training data with augmentation.
    
    Args:
        image_size (tuple): Target size for resizing images (height, width)
        
    Returns:
        torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_test_transform(image_size=(128, 128)):
    """
    Returns a torchvision transform for test/validation data.
    
    Args:
        image_size (tuple): Target size for resizing images (height, width)
        
    Returns:
        torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# Optional: quick test
if __name__ == "__main__":
    train_transform = get_train_transform()
    test_transform = get_test_transform()
    print("Train and test transforms are ready!")
