import os
import torch
import wandb
from torch.utils.data import DataLoader


class Trainer:
    """
    Trainer class for training a PyTorch model.
    Handles training loop, loss calculation, optimizer, accuracy,
    checkpoint saving, and resume training.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: str = "cuda",
        learning_rate: float = 1e-3,
        model_name: str = "model",
        checkpoint_dir: str = None,
        resume: bool = True
    ):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model_name = model_name
        self.start_epoch = 0

        # ----------------------------
        # Checkpoint directory
        # ----------------------------
        if checkpoint_dir is None:
            self.model_directory = "models"
        else:
            self.model_directory = checkpoint_dir

        os.makedirs(self.model_directory, exist_ok=True)

        # ----------------------------
        # Resume from checkpoint if exists
        # ----------------------------
        if resume:
            self.load_checkpoint()

    def train_one_epoch(self, epoch: int, log_every: int = 5):
        self.model.train()
        batch_losses = []
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(self.data_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            batch_losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log per-batch training metrics to W&B (if active)
            try:
                if wandb.run is not None:
                    # global step across epochs
                    try:
                        num_batches = len(self.data_loader)
                    except Exception:
                        num_batches = batch_idx + 1
                    global_step = (epoch - 1) * num_batches + batch_idx
                    running_acc = 100.0 * correct / total if total > 0 else 0.0
                    wandb.log({
                        "Training Loss": loss.item(),
                        "Training Accuracy": running_acc,
                        "Epoch": epoch,
                        "Batch": batch_idx
                    }, step=global_step)
            except Exception:
                pass

            if batch_idx % log_every == 0:
                print(f"[Epoch {epoch}] Batch {batch_idx}: Loss = {loss.item():.4f}")

        avg_loss = sum(batch_losses) / len(batch_losses)
        accuracy = 100.0 * correct / total

        print(f"[Epoch {epoch}] Average Loss: {avg_loss:.4f}")
        print(f"[Epoch {epoch}] Accuracy: {accuracy:.2f}%")

        return avg_loss, accuracy

    # ----------------------------
    # SAVE CHECKPOINT
    # ----------------------------
    def save_model(self, epoch: int):
        try:
            path = os.path.join(self.model_directory, f"{self.model_name}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                path
            )
            print(f"Checkpoint saved at: {path}")
            return path
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return None

    # ----------------------------
    # LOAD CHECKPOINT
    # ----------------------------
    def load_checkpoint(self):
        path = os.path.join(self.model_directory, f"{self.model_name}.pth")

        if not os.path.exists(path):
            print("No checkpoint found. Training from scratch.")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1

        print(f"Resumed training from epoch {self.start_epoch}")


# ----------------------------
# Quick local / Colab test
# ----------------------------
if __name__ == "__main__":
    from src.data.loader import get_dataloaders
    from src.model.cnn import CNN

    try:
        from google.colab import drive
        drive.mount("/content/drive")
        CHECKPOINT_DIR = "/content/drive/MyDrive/DrowsinessProject/artifacts/checkpoints"
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    except:
        CHECKPOINT_DIR = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, _ = get_dataloaders(batch_size=16, num_workers=1)
    model = CNN(num_classes=13)

    trainer = Trainer(
        model=model,
        data_loader=train_loader,
        device=device,
        model_name="drowsiness_cnn",
        checkpoint_dir=CHECKPOINT_DIR
    )

    EPOCHS = 10
    for epoch in range(trainer.start_epoch, EPOCHS):
        trainer.train_one_epoch(epoch)
        trainer.save_model(epoch)
