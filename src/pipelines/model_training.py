# src/pipelines/model_training.py
import sys
import os
from datetime import datetime

# ----------------------------
# Add project root to sys.path
# ----------------------------
PROJECT_ROOT = "/content/Drowsiness_Detection"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
import wandb
from src.data.loader import get_dataloaders
from src.model.cnn import CNN
from src.model.train import Trainer
from src.model.evaluation import Evaluator


def main():
    try:
        # ----------------------------
        # Paths (Google Drive already mounted manually)
        # ----------------------------
        BASE_DRIVE_PATH = "/content/drive/MyDrive/DrowsinessProject"
        os.makedirs(BASE_DRIVE_PATH, exist_ok=True)

        CHECKPOINT_DIR = os.path.join(BASE_DRIVE_PATH, "artifacts", "checkpoints")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        FINAL_MODEL_DIR = os.path.join(BASE_DRIVE_PATH, "artifacts", "model")
        os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

        # ----------------------------
        # Training configuration
        # ----------------------------
        EPOCHS = 10
        BATCH_SIZE = 32
        LEARNING_RATE = 0.001
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        NUM_CLASSES = 2  # Training on 2 classes: Closed and Opened
        CLASS_NAMES = ["Closed", "Opened"]  # Class 0: Closed (Drowsy), Class 1: Opened (Awake)

        config = {
            "Epochs": EPOCHS,
            "Batch Size": BATCH_SIZE,
            "Learning Rate": LEARNING_RATE,
            "Device": DEVICE,
            "Model": "CNN",
            "Num Classes": NUM_CLASSES,
            "Classes": CLASS_NAMES
        }

        # ----------------------------
        # Initialize W&B
        # ----------------------------
        wandb.init(
            project="Drowsiness-Detection-CNN",
            config=config,
            name=f"Experiment-{datetime.now().strftime('%d_%m_%Y_%H_%M')}"
        )

        # ----------------------------
        # Load data
        # ----------------------------
        train_loader, test_loader, class_names = get_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=2
        )
        # Determine number of batches to compute consistent global step values
        try:
            NUM_BATCHES = len(train_loader)
        except Exception:
            NUM_BATCHES = None
        
        # Auto-detect number of classes from data
        NUM_CLASSES = len(class_names)
        print(f"\n✅ Auto-detected {NUM_CLASSES} classes: {class_names}")
        print(f"   Class 0: {class_names[0]}")
        print(f"   Class 1: {class_names[1] if len(class_names) > 1 else 'N/A'}\n")

        # ----------------------------
        # Initialize model
        # ----------------------------
        model = CNN(num_classes=NUM_CLASSES).to(DEVICE)
        print(f"Using device: {DEVICE}")
        print(f"Model initialized with {NUM_CLASSES} classes")
        # Watch the model so W&B receives gradients/parameters and keeps live charts updated
        try:
            wandb.watch(model, log="all", log_freq=100)
        except Exception as e:
            print(f"W&B watch failed: {e}")

        # ----------------------------
        # Initialize Trainer & Evaluator
        # ----------------------------
        trainer = Trainer(
            model=model,
            data_loader=train_loader,
            device=DEVICE,
            learning_rate=LEARNING_RATE,
            model_name="drowsiness_cnn",
            checkpoint_dir=CHECKPOINT_DIR
        )

        evaluator = Evaluator(model=model, data_loader=test_loader, device=DEVICE)

        best_accuracy = 0.0

        # ----------------------------
        # Epoch loop
        # ----------------------------
        for epoch in range(1, EPOCHS + 1):
            # Training
            train_loss, train_acc = trainer.train_one_epoch(epoch=epoch)

            # Validation
            val_loss, val_acc = evaluator.evaluate(epoch=epoch)

            print(f"[Epoch {epoch}] Training Loss: {train_loss:.4f}, Training Acc: {train_acc:.2f}%")
            print(f"[Epoch {epoch}] Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%")

            # ----------------------------
            # Log ALL 4 metrics to W&B with clean names
            # ----------------------------
            metrics = {
                "Training Loss": train_loss,
                "Training Accuracy": train_acc,
                "Validation Loss": val_loss,
                "Validation Accuracy": val_acc,
                "epoch": epoch
            }
            # Use a global step consistent with per-batch logging so charts update live
            try:
                step_val = epoch * NUM_BATCHES if NUM_BATCHES else epoch
                wandb.log(metrics, step=step_val)
            except Exception:
                # fallback to default logging if wandb not available or step fails
                try:
                    wandb.log(metrics)
                except Exception:
                    pass
            print(f"✅ Logged to W&B - Epoch {epoch}\n")

            # ----------------------------
            # Save model to W&B ONLY (every epoch)
            # ----------------------------
            epoch_model_filename = f"drowsiness_epoch{epoch}_train{train_acc:.2f}_val{val_acc:.2f}.pth"
            torch.save({"model_state_dict": model.state_dict()}, epoch_model_filename)

            artifact = wandb.Artifact(
                name=f"drowsiness_epoch{epoch}_train{train_acc:.2f}_val{val_acc:.2f}",
                type="model"
            )
            artifact.add_file(epoch_model_filename)
            wandb.log_artifact(artifact)
            print(f"Epoch {epoch} model logged to W&B as artifact")

            # Remove the temporary local file to avoid clutter
            os.remove(epoch_model_filename)

            # ----------------------------
            # Save best model to Drive ONLY
            # ----------------------------
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_model_path = os.path.join(FINAL_MODEL_DIR, "drowsiness_cnn_best.pth")
                torch.save({"model_state_dict": model.state_dict()}, best_model_path)
                print(f"Best model updated at Epoch {epoch} with Validation Accuracy {val_acc:.2f}%")

        # ----------------------------
        # Save final model (optional) to W&B
        # ----------------------------
        final_model_path = os.path.join(FINAL_MODEL_DIR, "drowsiness_cnn_final.pth")
        torch.save({"model_state_dict": model.state_dict()}, final_model_path)
        print(f"Final model saved at: {final_model_path}")

        final_artifact = wandb.Artifact("drowsiness_cnn_final", type="model")
        final_artifact.add_file(final_model_path)
        wandb.log_artifact(final_artifact)

    except Exception as e:
        print(f"Error in Training Script: {e}")
        raise


if __name__ == "__main__":
    wandb.login(key=os.environ.get("WANDB_API_KEY", None))
    main()
