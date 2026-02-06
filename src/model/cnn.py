import torch
import torch.nn as nn
import torch.nn.functional as F

class DrowsinessCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DrowsinessCNN, self).__init__()
        
        # Conv layer 1: input 3 channels (RGB), output 16 channels, kernel size 3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        
        # Conv layer 2: input 16 channels, output 32 filters, kernel size 3x3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Conv layer 3: input 32 channels, output 64 filters, kernel size 3x3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # flattened feature map
        self.fc2 = nn.Linear(128, num_classes)   # output layer
    
    # Forward pass
    def forward(self, x):
        # Convolution + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 64 * 28 * 28)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Test the model
if __name__ == "__main__":
    model = DrowsinessCNN(num_classes=2)
    print(model)
    
    # Test with a dummy input (batch_size=4, 3 channels, 224x224 image)
    x = torch.randn(4, 3, 224, 224)
    outputs = model(x)
    print("Output shape:", outputs.shape)  # should be [4, 2]
