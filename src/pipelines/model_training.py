# src/pipelines/model_training.py
import sys
import os
from datetime import datetime
from pathlib import Path

# ----------------------------
# Add project root to sys.path (auto-detect)
# ----------------------------
# Get the project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROJECT_ROOT = str(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available. Training will continue without W&B logging.")

from src.data.loader import get_dataloaders
from src.model.cnn import CNN
from src.model.train import Trainer
from src.model.evaluation import Evaluator


def main():
    try:
        # ----------------------------
        # Paths (local or Colab)
        # ----------------------------
        # Try to detect if we're in Colab
        try:
            from google.colab import drive  # type: ignore
            # Running in Colab
            BASE_DRIVE_PATH = "/content/drive/MyDrive/DrowsinessProject"
            os.makedirs(BASE_DRIVE_PATH, exist_ok=True)
            CHECKPOINT_DIR = os.path.join(BASE_DRIVE_PATH, "artifacts", "checkpoints")
            FINAL_MODEL_DIR = os.path.join(BASE_DRIVE_PATH, "artifacts", "model")
        except ImportError:
            # Running locally - use project root
            BASE_PATH = os.path.join(PROJECT_ROOT, "artifacts")
            CHECKPOINT_DIR = os.path.join(BASE_PATH, "checkpoints")
            FINAL_MODEL_DIR = os.path.join(BASE_PATH, "models")
        
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
        
        print(f"📁 Checkpoint directory: {CHECKPOINT_DIR}")
        print(f"📁 Model directory: {FINAL_MODEL_DIR}")

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
        # Initialize W&B (optional)
        # ----------------------------
        if WANDB_AVAILABLE:
            try:
                wandb.init(
                    project="Drowsiness-Detection-CNN",
                    config=config,
                    name=f"Experiment-{datetime.now().strftime('%d_%m_%Y_%H_%M')}"
                )
                print("✅ W&B initialized")
            except Exception as e:
                print(f"⚠️  W&B initialization failed: {e}. Continuing without W&B.")
                WANDB_AVAILABLE = False
        else:
            print("ℹ️  Training without W&B logging")

        # ----------------------------
        # Load data with class weights calculation
        # ----------------------------
        train_loader, test_loader, class_names, class_weights = get_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=2,
            calculate_weights=True  # Calculate class weights for imbalanced datasets
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
        if WANDB_AVAILABLE:
            try:
                wandb.watch(model, log="all", log_freq=100)
            except Exception as e:
                print(f"W&B watch failed: {e}")

        # ----------------------------
        # Initialize Trainer & Evaluator with class weights
        # ----------------------------
        trainer = Trainer(
            model=model,
            data_loader=train_loader,
            device=DEVICE,
            learning_rate=LEARNING_RATE,
            model_name="drowsiness_cnn",
            checkpoint_dir=CHECKPOINT_DIR,
            class_weights=class_weights,  # Use class weights to handle imbalance
            use_scheduler=True  # Enable learning rate scheduling
        )

        evaluator = Evaluator(model=model, data_loader=test_loader, device=DEVICE)

        best_accuracy = 0.0
        best_val_loss = float('inf')
        patience = 5  # Early stopping patience
        patience_counter = 0

        # ----------------------------
        # Epoch loop with early stopping
        # ----------------------------
        for epoch in range(1, EPOCHS + 1):
            # Training
            train_loss, train_acc, train_class_acc = trainer.train_one_epoch(epoch=epoch)

            # Validation
            val_loss, val_acc, val_class_acc = evaluator.evaluate(epoch=epoch)

            # Update learning rate scheduler
            trainer.update_scheduler(val_loss)

            print(f"\n{'='*60}")
            print(f"[Epoch {epoch}] Training Loss: {train_loss:.4f}, Training Acc: {train_acc:.2f}%")
            print(f"[Epoch {epoch}] Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%")
            print(f"{'='*60}\n")

            # ----------------------------
            # Log metrics to W&B
            # ----------------------------
            metrics = {
                "Training Loss": train_loss,
                "Training Accuracy": train_acc,
                "Validation Loss": val_loss,
                "Validation Accuracy": val_acc,
                "Learning Rate": trainer.optimizer.param_groups[0]['lr'],
                "epoch": epoch
            }
            
            # Add per-class accuracies
            for class_idx, acc in train_class_acc.items():
                metrics[f"Train_Class_{class_idx}_Acc"] = acc
            for class_idx, acc in val_class_acc.items():
                metrics[f"Val_Class_{class_idx}_Acc"] = acc
            
            if WANDB_AVAILABLE:
                try:
                    step_val = epoch * NUM_BATCHES if NUM_BATCHES else epoch
                    wandb.log(metrics, step=step_val)
                    print(f"✅ Logged to W&B - Epoch {epoch}\n")
                except Exception as e:
                    print(f"⚠️  W&B logging failed: {e}\n")
            else:
                print(f"✅ Epoch {epoch} completed\n")

            # ----------------------------
            # Save checkpoint every epoch
            # ----------------------------
            is_best = val_acc > best_accuracy
            if is_best:
                best_accuracy = val_acc
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            metrics_dict = {
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "val_loss": val_loss
            }
            trainer.save_model(epoch, is_best=is_best, metrics=metrics_dict)

            # ----------------------------
            # Save model to W&B (every epoch, optional)
            # ----------------------------
            if WANDB_AVAILABLE:
                try:
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
                except Exception as e:
                    print(f"Warning: Could not save to W&B: {e}")

            # ----------------------------
            # Save best model to Drive
            # ----------------------------
            if is_best:
                best_model_path = os.path.join(FINAL_MODEL_DIR, "drowsiness_cnn_best.pth")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "train_loss": train_loss
                }, best_model_path)
                print(f"✅ Best model updated at Epoch {epoch} with Validation Accuracy {val_acc:.2f}%")
            
            # ----------------------------
            # Early stopping
            # ----------------------------
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered! No improvement for {patience} epochs.")
                print(f"   Best validation accuracy: {best_accuracy:.2f}%")
                break

        # ----------------------------
        # Save final model
        # ----------------------------
        final_model_path = os.path.join(FINAL_MODEL_DIR, "drowsiness_cnn_final.pth")
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": EPOCHS,
            "best_val_acc": best_accuracy,
            "config": config
        }, final_model_path)
        print(f"✅ Final model saved at: {final_model_path}")

        if WANDB_AVAILABLE:
            try:
                final_artifact = wandb.Artifact("drowsiness_cnn_final", type="model")
                final_artifact.add_file(final_model_path)
                wandb.log_artifact(final_artifact)
                print("✅ Final model logged to W&B")
            except Exception as e:
                print(f"⚠️  Could not log final model to W&B: {e}")
        
        print(f"\n🎉 Training completed!")
        print(f"📊 Best validation accuracy: {best_accuracy:.2f}%")
        print(f"💾 Best model saved at: {os.path.join(FINAL_MODEL_DIR, 'drowsiness_cnn_best.pth')}")

    except Exception as e:
        print(f"Error in Training Script: {e}")
        raise


if __name__ == "__main__":
    if WANDB_AVAILABLE:
        # Try to login if API key is available
        api_key = os.environ.get("WANDB_API_KEY", None)
        if api_key:
            try:
                wandb.login(key=api_key)
            except Exception as e:
                print(f"⚠️  W&B login failed: {e}. Continuing without W&B.")
    main()
