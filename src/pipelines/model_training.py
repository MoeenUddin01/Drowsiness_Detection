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


# ============================================================
# Save confusion matrix (NO warnings, always correct shape)
# ============================================================
def save_confusion_matrix(y_true, y_pred, epoch, artifact_root="artifacts"):
    cm_dir = os.path.join(artifact_root, "confusion_matrix")
    os.makedirs(cm_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Awake", "Drowsy"],
        yticklabels=["Awake", "Drowsy"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - Epoch {epoch}")

    cm_path = os.path.join(cm_dir, f"epoch_{epoch}_cm.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"Confusion matrix saved at: {cm_path}")


# ============================================================
# Training pipeline
# ============================================================
def run_training_pipeline(
    train_data_dir="datas/processed/train",
    test_data_dir="datas/processed/test",
    batch_size=16,
    learning_rate=0.001,
    num_epochs=20,
    model_name="drowsiness_cnn",
    device=None,
    artifact_root="artifacts",
):
    try:
        # 1️⃣ Device
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        # 2️⃣ Artifact folders
        models_dir = os.path.join(artifact_root, "models")
        checkpoints_dir = os.path.join(artifact_root, "checkpoints")
        cm_dir = os.path.join(artifact_root, "confusion_matrix")

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(cm_dir, exist_ok=True)

        # 3️⃣ W&B (NEW RUN EVERY TIME – NO CHART RESUME)
        wandb.init(
            project="Drowsiness-Detection-CNN",
            name=f"Experiment-{datetime.now().strftime('%d_%m_%Y_%H_%M')}",
            resume=False,
            reinit=True,
            config={
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "model": "DrowsinessCNN",
                "device": str(device),
            },
        )

        # 4️⃣ Datasets
        train_dataset = DrowsinessDataset(train_data_dir, train=True)
        val_dataset = DrowsinessDataset(test_data_dir, train=False)

        train_loader = get_dataloader(train_dataset, batch_size, shuffle=True)
        val_loader = get_dataloader(val_dataset, batch_size, shuffle=False)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

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
            device=device,
        )

        evaluator = Evaluator(
            batch_size=batch_size,
            data=val_loader,
            model=model,
            device=device,
        )

        # 7️⃣ Resume checkpoint (MODEL ONLY, NOT W&B)
        latest_ckpt = os.path.join(checkpoints_dir, "latest_checkpoint.pth")
        start_epoch = 1
        BEST_ACCURACY = 0.0

        if os.path.exists(latest_ckpt):
            ckpt = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            BEST_ACCURACY = ckpt.get("best_accuracy", 0.0)
            print(f"Resuming training from epoch {start_epoch}")

        # 8️⃣ Epoch loop
        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\n========== Epoch {epoch} ==========")

            # ---- Training ----
            train_out = trainer.start_training_loop(epoch)
            if train_out is None:
                print("⚠ Training failed, skipping epoch")
                continue

            avg_train_loss, _, train_acc = train_out

            # ---- Validation ----
            avg_val_loss, val_acc = evaluator.start_evaluation_loop(epoch)

            # ---- Confusion Matrix ----
            all_labels, all_preds = [], []

            model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    _, preds = torch.max(outputs, 1)
                    all_labels.extend(y.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

            save_confusion_matrix(all_labels, all_preds, epoch, artifact_root)

            # ---- W&B Log ----
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                }
            )

            # ---- Save checkpoint ----
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "best_accuracy": BEST_ACCURACY,
            }

            torch.save(ckpt, os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch}.pth"))
            torch.save(ckpt, latest_ckpt)
            print(f"Checkpoint saved for epoch {epoch}")

            # ---- Save best model ----
            if val_acc > BEST_ACCURACY:
                BEST_ACCURACY = val_acc
                best_model_path = os.path.join(models_dir, f"{model_name}_best.pth")
                torch.save(model.state_dict(), best_model_path)
                wandb.save(best_model_path)
                print(f"🔥 Best model saved (Accuracy: {val_acc:.2f}%)")

        print("Training completed successfully!")
        wandb.finish()

    except Exception as e:
        print(f"❌ Error in training pipeline: {e}")
        raise e


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    wandb.login(key=os.environ.get("WANDB_API_KEY", None))

    run_training_pipeline(
        batch_size=16,
        learning_rate=0.001,
        num_epochs=20,
        model_name="drowsiness_cnn",
        artifact_root="artifacts",
    )
