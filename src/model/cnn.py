# src/models/cnn.py

import torch
from torch import nn


class CNN(nn.Module):
    """
    Simple CNN for image classification.

    Input: 128x128 RGB images
    Architecture:
        Conv2d -> ReLU -> MaxPool2d
        Conv2d -> ReLU -> MaxPool2d
        Flatten -> Fully connected layers
    Output: number of classes (default 13)
    """

    def __init__(self, num_classes=13, input_channels=3):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 32 * 128, 256),  # After two 2x2 poolings, image size: 128 -> 64 -> 32
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


if __name__ == "__main__":
    # Quick test
    model = CNN(num_classes=13)
    x = torch.randn(1, 3, 128, 128)  # batch_size=1
    out = model(x)
    print(f"Output shape: {out.shape}")  # Should be [1, 13]
