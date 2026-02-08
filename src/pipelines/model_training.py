# src/pipelines/model_training.py
import sys
import os

# Add your project root to Python path
PROJECT_ROOT = "/content/Drowsiness_Detection"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import os
from datetime import datetime


import torch
import wandb
from src.data.dataloader import get_dataloaders
from src.models.cnn import CNN
from src.models.trainer import Trainer
from src.models.evaluator import Evaluator



def main():
    try:
        # ----------------------------
        # Google Drive paths (Drive already mounted manually)
        # ----------------------------
        BASE_DRIVE_PATH = "/content/drive/MyDrive/DrowsinessProject"
        os.makedirs(BASE_DRIVE_PATH, exist_ok=True)

        # Folder for checkpoints (during training)
        CHECKPOINT_DIR = os.path.join(BASE_DRIVE_PATH, "artifacts", "checkpoints")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        # Folder for final model (after training)
        FINAL_MODEL_DIR = os.path.join(BASE_DRIVE_PATH, "artifacts", "model")
        os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

        # ----------------------------
        # Training configuration
        # ----------------------------
        EPOCHS = 100
        BATCH_SIZE = 32
        LEARNING_RATE = 0.001
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        config = {
            "Epochs": EPOCHS,
            "Batch Size": BATCH_SIZE,
            "Learning Rate": LEARNING_RATE,
            "Device": DEVICE,
            "Model": "CNN"
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
        train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, num_workers=4)

        # ----------------------------
        # Initialize model
        # ----------------------------
        model = CNN(num_classes=13).to(DEVICE)
        print("Using device:", DEVICE)

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
            train_loss, _, train_acc = trainer.train_one_epoch(epoch=epoch)

            # Validation
            val_loss, _, val_acc = evaluator.evaluate(epoch=epoch)

            # Log metrics to W&B
            wandb.log({
                "Training Loss": train_loss,
                "Validation Loss": val_loss,
                "Training Accuracy": train_acc,
                "Validation Accuracy": val_acc,
                "Epoch": epoch
            })

            # Save best checkpoint during training
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                checkpoint_path = trainer.save_model()
                if checkpoint_path:
                    print(f"Best model checkpoint saved at Epoch {epoch} with Accuracy {val_acc:.2f}%")
                    # Optional: log checkpoint to W&B
                    artifact = wandb.Artifact("drowsiness_cnn_checkpoint", type="model")
                    artifact.add_file(checkpoint_path)
                    wandb.log_artifact(artifact)

        # ----------------------------
        # Save final model after all epochs
        # ----------------------------
        final_model_path = os.path.join(FINAL_MODEL_DIR, "drowsiness_cnn_final.pth")
        torch.save({"model_state_dict": model.state_dict()}, final_model_path)
        print(f"Final model saved at: {final_model_path}")

        # Optional: log final model to W&B
        final_artifact = wandb.Artifact("drowsiness_cnn_final", type="model")
        final_artifact.add_file(final_model_path)
        wandb.log_artifact(final_artifact)

    except Exception as e:
        print(f"Error in Training Script: {e}")
        raise


if __name__ == "__main__":
    wandb.login(key=os.environ.get("WANDB_API_KEY", None))
    main()
