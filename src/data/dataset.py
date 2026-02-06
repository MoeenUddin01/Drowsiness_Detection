#src/data/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
from .transforms import get_data_transforms

class DrowsinessDataset(Dataset):
    def __init__(self,data_dir,train=True):
        self.data_dir =data_dir
        self.train = train
        self.transform = get_data_transforms(train)
        
        #list to hold (image_path,label) tuples
        self.samples = []
        
        #map class names to labels
        self.class_to_label = {"closed_1":0,"opened_1":1}
        
        def _load_images(self):
            
        # Iterate over class folders
            for class_name, label in self.class_to_label.items():
                class_folder = os.path.join(self.data_dir, class_name)
            
                # Get all files in the folder
                for filename in os.listdir(class_folder):
                    if filename.endswith((".jpg", ".png", ".jpeg")):
                        # Store full path + label
                        self.samples.append((os.path.join(class_folder, filename), label))

        # Load images and labels
        def __len__(self):
            #return total number of images
            return len(self.samples)
        
        def __getitem__(self,idx):
            #get image path and label
            img_path,label = self.images[idx]
            #open image
            image=image.open(img_path).convert("RGB")
            #apply transforms
            if self.transform:
                image = self.transform(image)
                
            return image,label