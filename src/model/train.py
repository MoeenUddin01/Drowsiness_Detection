import torch
from torch.utils.data import DataLoader
import os

class Trainer:
    def __init__(self, batch_size:int, learning_rate:float, data:DataLoader, model, model_path:str, device:str):
        self.batch_size = batch_size
        self.data = data
        self.model = model
        self.lr = learning_rate
        self.device = device
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model_directory = "artifacts/models"
        os.makedirs(self.model_directory, exist_ok=True)
        self.model_path = model_path

    def train_epoch(self, epoch:int):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (x, y) in enumerate(self.data):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)  # raw logits, do NOT apply softmax
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)  # accumulate weighted by batch size
            _, predicted = torch.max(logits, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

            if batch_idx % 5 == 0:
                print(f"Training-> Epoch {epoch} -> Batch {batch_idx}: Loss={loss.item():.4f}")

        avg_loss = total_loss / total
        acc = 100.0 * correct / total if total > 0 else 0.0
        print(f"Epoch {epoch} -> Avg Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
        return avg_loss, acc

    def save_model(self):
        path = os.path.join(self.model_directory, f"{self.model_path}.pth")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved at {path}")
        return path
