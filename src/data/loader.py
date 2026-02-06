#src/data/loader.py

from torch.utils.data import DataLoader

def get_dataloader(dataset,batch_size=32,shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2
    )
    
    