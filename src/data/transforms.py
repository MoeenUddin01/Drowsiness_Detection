# # src/data/transforms.py

# from torchvision import transforms

# def get_train_transform(image_size=(128, 128)):
#     """
#     Returns a torchvision transform for training data with augmentation.
#     Enhanced with more augmentation techniques for better generalization.
    
#     Args:
#         image_size (tuple): Target size for resizing images (height, width)
        
#     Returns:
#         torchvision.transforms.Compose object
#     """
#     return transforms.Compose([
#         transforms.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),  # Slightly larger for cropping
#         transforms.RandomCrop(image_size),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomRotation(degrees=10),  # Small rotations
#         transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
#         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small translations
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#         transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))  # Random erasing for regularization
#     ])


# def get_test_transform(image_size=(128, 128)):
#     """
#     Returns a torchvision transform for test/validation data.
    
#     Args:
#         image_size (tuple): Target size for resizing images (height, width)
        
#     Returns:
#         torchvision.transforms.Compose object
#     """
#     return transforms.Compose([
#         transforms.Resize(image_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])


# # Optional: quick test
# if __name__ == "__main__":
#     train_transform = get_train_transform()
#     test_transform = get_test_transform()
#     print("Train and test transforms are ready!")
