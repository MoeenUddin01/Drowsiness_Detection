# src/models/cnn.py

import torch
from torch import nn


class CNN(nn.Module):
    """
    Simple CNN for Binary Classification (Closed vs Open Eyes)

    Input: 128x128 RGB images
    Output:
        0 -> closed
        1 -> open
    """

    def __init__(self, num_classes=2, input_channels=3):
        super(CNN, self).__init__()

        # -------------------------
        # Convolutional Block
        # -------------------------
        self.conv_block = nn.Sequential(

            # First Conv Layer
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128 -> 64

            # Second Conv Layer
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 64 -> 32
        )

        # -------------------------
        # Flatten Layer
        # -------------------------
        self.flatten = nn.Flatten()

        # -------------------------
        # Fully Connected Layers
        # -------------------------
        self.fc_layers = nn.Sequential(

            # After pooling:
            # 128x128 -> 64x64 -> 32x32
            # Channels = 128
            nn.Linear(32 * 32 * 128, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            # Final Output Layer (2 classes)
            nn.Linear(128, num_classes)
        )

    # -------------------------
    # Forward Pass
    # -------------------------
    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


# -------------------------
# Quick Test
# -------------------------
if __name__ == "__main__":
    model = CNN(num_classes=2)
    x = torch.randn(1, 3, 128, 128)
    output = model(x)
    print("Output shape:", output.shape)  # Should be [1, 2]
