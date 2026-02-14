import os
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter


class Trainer:
    """
    Trainer class for training a PyTorch model.
    Handles training loop, loss calculation, optimizer, accuracy,
    checkpoint saving, and resume training.
    Now includes class weights, learning rate scheduling, and per-class metrics.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: str = "cuda",
        learning_rate: float = 1e-3,
        model_name: str = "model",
        checkpoint_dir: str = None,
        resume: bool = True,
        class_weights: torch.Tensor = None,
        use_scheduler: bool = True
    ):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.model_name = model_name
        self.start_epoch = 0
        self.use_scheduler = use_scheduler

        # ----------------------------
        # Loss function with class weights (handles imbalance)
        # ----------------------------
        if class_weights is not None:
            class_weights = class_weights.to(device)
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            print(f"✅ Using class weights: {class_weights.cpu().numpy()}")
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
            print("⚠️  No class weights provided - model may be biased if classes are imbalanced")

        # ----------------------------
        # Optimizer with weight decay for regularization
        # ----------------------------
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-4  # L2 regularization
        )

        # ----------------------------
        # Learning rate scheduler
        # ----------------------------
        if use_scheduler:
            # Create scheduler without verbose (not supported in all PyTorch versions)
            # We'll print LR changes manually in update_scheduler method
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=3
            )
        else:
            self.scheduler = None

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
        
        # Per-class tracking
        class_correct = Counter()
        class_total = Counter()

        for batch_idx, (inputs, labels) in enumerate(self.data_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            batch_losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Track per-class accuracy
            for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

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
                        "Learning Rate": self.optimizer.param_groups[0]['lr'],
                        "Epoch": epoch,
                        "Batch": batch_idx
                    }, step=global_step)
            except Exception:
                pass

            if batch_idx % log_every == 0:
                print(f"[Epoch {epoch}] Batch {batch_idx}: Loss = {loss.item():.4f}, LR = {self.optimizer.param_groups[0]['lr']:.6f}")

        avg_loss = sum(batch_losses) / len(batch_losses)
        accuracy = 100.0 * correct / total

        # Calculate per-class accuracies
        class_accuracies = {}
        for class_idx in class_total.keys():
            class_acc = 100.0 * class_correct[class_idx] / class_total[class_idx] if class_total[class_idx] > 0 else 0.0
            class_accuracies[class_idx] = class_acc

        print(f"[Epoch {epoch}] Average Loss: {avg_loss:.4f}")
        print(f"[Epoch {epoch}] Overall Accuracy: {accuracy:.2f}%")
        for class_idx, acc in sorted(class_accuracies.items()):
            print(f"  Class {class_idx} Accuracy: {acc:.2f}% ({class_correct[class_idx]}/{class_total[class_idx]})")

        return avg_loss, accuracy, class_accuracies

    # ----------------------------
    # SAVE CHECKPOINT
    # ----------------------------
    def save_model(self, epoch: int, is_best: bool = False, metrics: dict = None):
        try:
            # Regular checkpoint
            path = os.path.join(self.model_directory, f"{self.model_name}.pth")
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            if self.scheduler is not None:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            if metrics:
                checkpoint["metrics"] = metrics
                
            torch.save(checkpoint, path)
            print(f"Checkpoint saved at: {path}")
            
            # Save best model separately
            if is_best:
                best_path = os.path.join(self.model_directory, f"{self.model_name}_best.pth")
                torch.save(checkpoint, best_path)
                print(f"✅ Best model saved at: {best_path}")
                return best_path
            
            return path
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return None
    
    def update_scheduler(self, val_loss: float):
        """Update learning rate scheduler based on validation loss"""
        if self.scheduler is not None:
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"📉 Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

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
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"  Scheduler state restored")

        print(f"Resumed training from epoch {self.start_epoch}")


# ----------------------------
# Quick local / Colab test
# ----------------------------
if __name__ == "__main__":
    from src.data.loader import get_dataloaders
    from src.model.cnn import CNN
    from src.model.evaluation import Evaluator

    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
        CHECKPOINT_DIR = "/content/drive/MyDrive/DrowsinessProject/artifacts/checkpoints"
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    except ImportError:
        CHECKPOINT_DIR = "artifacts/checkpoints"
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader, class_names, class_weights = get_dataloaders(
        batch_size=16, 
        num_workers=1,
        calculate_weights=True
    )
    
    num_classes = len(class_names)
    model = CNN(num_classes=num_classes)

    trainer = Trainer(
        model=model,
        data_loader=train_loader,
        device=device,
        model_name="drowsiness_cnn",
        checkpoint_dir=CHECKPOINT_DIR,
        class_weights=class_weights,
        use_scheduler=True
    )
    
    evaluator = Evaluator(model=model, data_loader=test_loader, device=device)

    EPOCHS = 10
    best_val_acc = 0.0
    
    for epoch in range(trainer.start_epoch, EPOCHS):
        train_loss, train_acc, _ = trainer.train_one_epoch(epoch)
        val_loss, val_acc, _ = evaluator.evaluate(epoch)
        trainer.update_scheduler(val_loss)
        
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        trainer.save_model(epoch, is_best=is_best)