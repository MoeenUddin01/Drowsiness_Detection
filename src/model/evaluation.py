# src/models/evaluator.py

import torch
import wandb
from torch.utils.data import DataLoader


class Evaluator:
    """
    Evaluates a PyTorch model on a given dataset.

    Computes average validation loss and accuracy for one epoch.
    """

    def __init__(self, model: torch.nn.Module, data_loader: DataLoader, device: str = "cuda"):
        """
        Args:
            model: The PyTorch model to evaluate.
            data_loader: DataLoader for validation/test dataset.
            device: Device to run evaluation on ("cuda" or "cpu").
        """
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()

    def evaluate(self, epoch: int = 0, log_every: int = 5):
        """
        Run evaluation loop for one epoch.

        Args:
            epoch: Current epoch number (for logging purposes).
            log_every: Print loss every N batches.

        Returns:
            avg_loss: Average validation loss for the epoch (float)
            accuracy: Accuracy (%) for the dataset (float)
        """
        self.model.eval()
        correct = 0
        total = 0
        batch_losses = []

        try:
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(self.data_loader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    batch_losses.append(loss.item())

                    # Calculate accuracy
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    if batch_idx % log_every == 0:
                        print(f"[Epoch {epoch}] Batch {batch_idx}: Loss = {loss.item():.4f}")

            # Compute average loss and accuracy
            avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
            accuracy = 100.0 * correct / total if total > 0 else 0.0

            # Log validation metrics to W&B (per-epoch)
            try:
                if wandb.run is not None:
                    wandb.log({
                        "Validation Loss": avg_loss,
                        "Validation Accuracy": accuracy,
                        "Epoch": epoch
                    }, step=epoch)
            except Exception:
                pass

            print(f"[Epoch {epoch}] Average Validation Loss: {avg_loss:.4f}")
            print(f"[Epoch {epoch}] Validation Accuracy: {accuracy:.2f}%")

            return avg_loss, accuracy  # only return floats now

        except Exception as e:
            print(f"Error during evaluation at Epoch {epoch}, Batch {batch_idx}: {e}")
            return 0.0, 0.0


# ----------------------------
# Quick local / Colab test
# ----------------------------
if __name__ == "__main__":
    from src.data.loader import get_dataloaders
    from src.model.cnn import CNN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_loader = get_dataloaders(batch_size=16, num_workers=1)
    model = CNN(num_classes=13).to(device)

    evaluator = Evaluator(model=model, data_loader=test_loader, device=device)
    val_loss, val_acc = evaluator.evaluate(epoch=1)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
