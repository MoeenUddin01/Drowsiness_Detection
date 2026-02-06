#src/data/transforms.py
from torchvision import transforms

def get_data_transforms(train=True):
    if train:
        return transforms.Compose([
            trasnform.Resize((124, 124)),
            transforms.randomHorizontalFlip(),
            trainsforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]       
                
            )   ])
    else:
        return transforms.Compose([
            transforms.Resize((124, 124)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        