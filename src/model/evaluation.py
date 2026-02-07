import torch
from torch.utils.data import DataLoader
from src.data.dataset import DrowsinessDataset
from src.data.loader import get_dataloader
from src.model.cnn import DrowsinessCNN
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class Evaluator:
    """
    Evaluation loop for validation/testing.
    Computes:
    - Validation loss
    - Validation accuracy
    - Confusion matrix (saved as image)
    """

    def __init__(self, batch_size: int, data: DataLoader, model, device):
        self.batch_size = batch_size
        self.data = data
        self.model = model
        self.device = device
        self.loss = torch.nn.CrossEntropyLoss()

    def start_evaluation_loop(self, epoch=1):
        try:
            self.model.eval()

            correct = 0
            total = 0
            validation_losses = []

            # Store predictions & labels for confusion matrix
            all_preds = []
            all_labels = []

            for batch, (x, y) in enumerate(self.data):
                with torch.no_grad():
                    x = x.to(self.device)
                    y = y.to(self.device)

                    outputs = self.model(x)
                    val_loss = self.loss(outputs, y)
                    validation_losses.append(val_loss.item())

                    _, predicted = torch.max(outputs, 1)

                    total += y.size(0)
                    correct += (predicted == y).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

                    if batch % 5 == 0:
                        print(
                            f"Validation-> Epoch {epoch} -> Batch {batch}: "
                            f"Loss={val_loss.item():.4f}"
                        )

            avg_val_loss = (
                sum(validation_losses) / len(validation_losses)
                if validation_losses else 0.0
            )

            val_acc = 100.0 * correct / total if total > 0 else 0.0

            print(f"Average Epoch Validation Loss {epoch}: {avg_val_loss:.4f}")
            print(f"Validation Accuracy Epoch {epoch}: {val_acc:.2f}%")

            # ===============================
            # Save Confusion Matrix
            # ===============================
            artifact_root = "artifacts"
            cm_dir = os.path.join(artifact_root, "confusion_matrix")
            os.makedirs(cm_dir, exist_ok=True)

            cm = confusion_matrix(all_labels, all_preds)

            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Alert", "Drowsy"],
                yticklabels=["Alert", "Drowsy"]
            )
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix - Epoch {epoch}")

            cm_path = os.path.join(cm_dir, f"epoch_{epoch}_cm.png")
            plt.savefig(cm_path)
            plt.close()

            print(f"Confusion matrix saved at: {cm_path}")

            return avg_val_loss, val_acc

        except Exception as e:
            print(f"Error in Validation Loop (Epoch {epoch}): {e}")
            raise e


# ===============================
# Standalone Test (Optional)
# ===============================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = DrowsinessDataset(
        data_dir="datas/processed/test",
        train=False
    )

    test_loader = get_dataloader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )

    model = DrowsinessCNN(num_classes=2).to(device)
    model_path = "artifacts/models/drowsiness_cnn.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    evaluator = Evaluator(
        batch_size=16,
        data=test_loader,
        model=model,
        device=device
    )

    evaluator.start_evaluation_loop(epoch=1)
