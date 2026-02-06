import torch
import torch.nn as nn
import torch.nn.functional as F

class DrowsinessCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DrowsinessCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # ✅ Correct for 128×128 input
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # ✅ SAFE flatten (this fixes your bug)
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# ✅ Correct test
if __name__ == "__main__":
    model = DrowsinessCNN(num_classes=2)
    print(model)

    # Match transform size (128×128)
    x = torch.randn(4, 3, 128, 128)
    outputs = model(x)
    print("Output shape:", outputs.shape)  # [4, 2]
