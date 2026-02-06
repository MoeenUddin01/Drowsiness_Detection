import torch
from torch.utils.data import DataLoader
from src.data.loader import get_dataloader
from src.data.dataset import DrowsinessDataset
from src.model.cnn import DrowsinessCNN
from src.model.train import Trainer
from src.model.evaluaion import Evaluator
import wandb
import os
from datetime import datetime

def run_training_pipeline(
    train_data_dir="datas/processed/train",
    test_data_dir="datas/processed/test",
    batch_size=16,
    learning_rate=0.001,
    num_epochs=10,
    model_name="drowsiness_cnn",
    device=None,
    artifact_root="artifacts",  # Can be set to Google Drive path in Colab
              
):
    try:
        # 1️⃣ Device setup
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        # 2️⃣ Create artifact folders
        os.makedirs(os.path.join(artifact_root, "models"), exist_ok=True)
        os.makedirs(os.path.join(artifact_root, "checkpoints"), exist_ok=True)

        # 3️⃣ Initialize W&B
        config = {
            "Epochs": num_epochs,
            "Batch Size": batch_size,
            "Learning Rate": learning_rate,
            "Device": str(device),
            "Model": "DrowsinessCNN"
        }
        #run_id=""
        wandb.init(
            project="Drowsiness-Detection-CNN",
            config=config,
            #id=run_id,
            #resume="allow",
            name=f'Experiment-{datetime.now().strftime("%d_%m_%Y_%H_%M")}'
        
        )

        # 4️⃣ Load datasets
        train_dataset = DrowsinessDataset(data_dir=train_data_dir, train=True)
        train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = DrowsinessDataset(data_dir=test_data_dir, train=False)
        test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

        # 5️⃣ Initialize model
        model = DrowsinessCNN(num_classes=2).to(device)
        print("Model initialized")

        # 6️⃣ Initialize Trainer and Evaluator
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
            data=test_loader,
            model=model,
            device=device
        )

        # 7️⃣ Check for latest checkpoint to resume
        checkpoint_dir = os.path.join(artifact_root, "checkpoints")
        latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")

        start_epoch = 1
        BEST_ACCURACY = 0

        if os.path.exists(latest_checkpoint_path):
            checkpoint = torch.load(latest_checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            BEST_ACCURACY = checkpoint.get("best_accuracy", 0)
            print(f"Resuming training from epoch {start_epoch}, best accuracy so far: {BEST_ACCURACY:.2f}%")

        # 8️⃣ Epoch loop
        for epoch in range(start_epoch, num_epochs + 1):
            # Training
            avg_train_loss, _, train_acc = trainer.start_training_loop(epoch)

            # Evaluation
            avg_val_loss, _, val_acc = evaluator.start_evaluation_loop(epoch)

            # Log metrics to W&B
            wandb.log({
                "Epoch": epoch,
                "Training Loss": avg_train_loss,
                "Validation Loss": avg_val_loss,
                "Training Accuracy": train_acc,
                "Validation Accuracy": val_acc
            })

            # Save checkpoint every epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "best_accuracy": BEST_ACCURACY
            }
            epoch_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(checkpoint, epoch_checkpoint_path)
            torch.save(checkpoint, latest_checkpoint_path)
            print(f"Checkpoint saved at {epoch_checkpoint_path}")

            # Save best model
            if val_acc > BEST_ACCURACY:
                BEST_ACCURACY = val_acc
                final_model_path = os.path.join(artifact_root, "models", f"{model_name}.pth")
                torch.save(model.state_dict(), final_model_path)
                wandb.save(final_model_path)
                print(f"Best model with Accuracy {val_acc:.2f}% saved at {final_model_path}")

        print("Training pipeline completed successfully!")

    except Exception as e:
        print(f"Error in training pipeline: {e}")
        raise e


# ===============================
# Run pipeline
# ===============================
if __name__ == "__main__":
    # Login to W&B (API key from environment variable)
    import os
    wandb.login(key=os.environ.get("WANDB_API_KEY", None))

    # Example: if using Colab, mount Google Drive first and set artifact_root
    # artifact_root = "/content/drive/MyDrive/Drowsiness_Project/artifacts"
    
    run_training_pipeline(
        batch_size=16,
        learning_rate=0.001,
        num_epochs=20,              # Can increase epochs; resumes from last checkpoint
        model_name="drowsiness_cnn",
        artifact_root="artifacts",  # Change to Drive path in Colab
        #wandb_run_id=None            # Add your run ID to continue in same W&B chart
    )
