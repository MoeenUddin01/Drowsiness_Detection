import torch
from torch.utils.data import DataLoader
from src.data.dataset import DrowsinessDataset
from src.data.loader import get_dataloader
from src.model.cnn import DrowsinessCNN
import os

# ===============================
# Trainer Class
# ===============================
class Trainer:
    """
    Encapsulates training loop, loss, optimizer, device handling, and model saving
    """
    def __init__(self, batch_size:int, learning_rate:float, data: DataLoader, model, model_path:str, device:str):
        self.batch_size = batch_size
        self.data = data
        self.model = model
        self.lr = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Save models to artifacts/models folder
        self.model_directory = "artifacts/models"
        os.makedirs(self.model_directory, exist_ok=True)
        self.model_path = model_path
        self.device = device
    
    def start_training_loop(self, epoch:int):
        try:
            self.model.train()  # enable training mode
            training_losses = []
            correct = 0
            total = 0
            
            for batch, (x, y) in enumerate(self.data):
                # Move to device
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                prediction = self.model(x)
                training_loss = self.loss(prediction, y)
                training_losses.append(training_loss.item())
                
                # Backward pass
                self.optimizer.zero_grad()
                training_loss.backward()
                self.optimizer.step()
                
                # Accuracy calculation
                _, predicted = torch.max(prediction.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
                if batch % 5 == 0:
                    print(f"Training-> Epoch {epoch} -> Batch {batch}: Loss={training_loss.item():.4f}")
            
            # Epoch stats
            epoch_acc = 100.0 * correct / total if total > 0 else 0.0
            average_epoch_training_loss = sum(training_losses) / len(training_losses) if training_losses else 0.0
            
            print(f"Epoch {epoch} -> Average Loss: {average_epoch_training_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
            
            return average_epoch_training_loss, training_losses, epoch_acc
        
        except Exception as e:
            print(f"Error in Training Loop Epoch {epoch}, Batch {batch}: {e}")
            return None
    
    def save_model(self):
        try:
            final_path = os.path.join(self.model_directory, f"{self.model_path}.pth")
            torch.save(
                {"model_state_dict": self.model.state_dict()},
                final_path
            )
            print(f"Model saved at {final_path}")
            return final_path
        except Exception as e:
            print(f"Error saving model: {e}")
            return None

# ===============================
# MAIN TRAINING SCRIPT
# ===============================
if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Load datasets
    train_dataset = DrowsinessDataset(data_dir="datas/processed/train", train=True)
    train_loader = get_dataloader(train_dataset, batch_size=16, shuffle=True)
    
    # Initialize model
    model = DrowsinessCNN(num_classes=2).to(device)
    
    # Initialize Trainer
    trainer = Trainer(
        batch_size=16,
        learning_rate=0.001,
        data=train_loader,
        model=model,
        model_path="drowsiness_cnn",
        device=device
    )
    
    # Training loop
    num_epochs = 10
    for epoch in range(1, num_epochs+1):
        trainer.start_training_loop(epoch)
    
    # Save the trained model
    trainer.save_model()
