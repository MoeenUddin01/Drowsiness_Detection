import torch
from torch.utils.data import DataLoader
from src.data.loader import get_dataloader
from src.data.dataset import DrowsinessDataset
from src.model.cnn import DrowsinessCNN
from src.model.train import Trainer
from src.model.evaluation import Evaluator
import wandb
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def save_confusion_matrix(y_true, y_pred, epoch, artifact_root="artifacts"):
    # 1️⃣ Create folder
    cm_dir = os.path.join(artifact_root, "confusion_matrix")
    os.makedirs(cm_dir, exist_ok=True)

    # 2️⃣ Compute matrix
    cm = confusion_matrix(y_true, y_pred)

    # 3️⃣ Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix Epoch {epoch}")

    # 4️⃣ Save figure
    cm_path = os.path.join(cm_dir, f"epoch_{epoch}_cm.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved: {cm_path}")


def run_training_pipeline(
    train_data_dir="datas/processed/train",
    test_data_dir="datas/processed/test",
    batch_size=16,
    learning_rate=0.001,
    num_epochs=20,
    model_name="drowsiness_cnn",
    device=None,
    artifact_root="artifacts"
):
    try:
        # 1️⃣ Device
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        # 2️⃣ Artifact folders
        models_dir = os.path.join(artifact_root, "models")
        checkpoints_dir = os.path.join(artifact_root, "checkpoints")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(os.path.join(artifact_root, "confusion_matrix"), exist_ok=True)

        # 3️⃣ W&B
        wandb.init(
            project="Drowsiness-Detection-CNN",
            name=f'Experiment-{datetime.now().strftime("%d_%m_%Y_%H_%M")}',
            config={
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "model": "DrowsinessCNN",
                "device": str(device)
            }
        )

        # 4️⃣ Datasets
        train_dataset = DrowsinessDataset(data_dir=train_data_dir, train=True)
        val_dataset = DrowsinessDataset(data_dir=test_data_dir, train=False)
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

        # 5️⃣ Model
        model = DrowsinessCNN(num_classes=2).to(device)
        print("Model initialized")

        # 6️⃣ Trainer & Evaluator
        trainer = Trainer(
            batch_size=batch_size,
            learning_rate=learning_rate,
            data=train_loader,
            model=model,
            model_path=model_name,
            device=device
        )
        evaluator = Evaluator(
            batch_size=batch_size,
            data=val_loader,
            model=model,
            device=device
        )

        # 7️⃣ Resume checkpoint
        latest_checkpoint_path = os.path.join(checkpoints_dir, "latest_checkpoint.pth")
        start_epoch = 1
        BEST_ACCURACY = 0

        if os.path.exists(latest_checkpoint_path):
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            BEST_ACCURACY = checkpoint.get("best_accuracy", 0)
            print(f"Resuming from epoch {start_epoch}, best accuracy: {BEST_ACCURACY:.2f}%")

        # 8️⃣ Epoch loop
        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\n========== Epoch {epoch} ==========")

            # ---- Training ----
            avg_train_loss, _, train_acc = trainer.start_training_loop(epoch)

            # ---- Validation ----
            avg_val_loss, val_acc = evaluator.start_evaluation_loop(epoch)

            # ---- Confusion matrix ----
            all_labels = []
            all_preds = []
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                with torch.no_grad():
                    preds = model(x)
                    _, predicted = torch.max(preds, 1)
                all_labels.extend(y.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
            save_confusion_matrix(all_labels, all_preds, epoch, artifact_root)

            # ---- W&B log ----
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc
            })

            # ---- Checkpoint ----
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "best_accuracy": BEST_ACCURACY
            }
            torch.save(checkpoint, os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch}.pth"))
            torch.save(checkpoint, latest_checkpoint_path)
            print(f"Checkpoint saved for epoch {epoch}")

            # ---- Best model ----
            if val_acc > BEST_ACCURACY:
                BEST_ACCURACY = val_acc
                final_model_path = os.path.join(models_dir, f"{model_name}_best.pth")
                torch.save(model.state_dict(), final_model_path)
                wandb.save(final_model_path)
                print(f"🔥 Best model updated (Accuracy: {val_acc:.2f}%) -> {final_model_path}")

        print("Training pipeline completed successfully!")
        wandb.finish()

    except Exception as e:
        print(f"Error in training pipeline: {e}")
        raise e


# ===============================
# Run pipeline
# ===============================
if __name__ == "__main__":
    wandb.login(key=os.environ.get("WANDB_API_KEY", None))

    run_training_pipeline(
        batch_size=16,
        learning_rate=0.001,
        num_epochs=20,
        model_name="drowsiness_cnn",
        artifact_root="artifacts"
    )
