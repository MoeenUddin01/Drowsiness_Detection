import torch
from torch.utils.data import DataLoader
from src.data.dataset import DrowsinessDataset
from src.data.loader import get_dataloader
from src.model.cnn import DrowsinessCNN
import os

class Evaluator:
    # Evaluation loop
    def __init__(self, batch_size:int, data: DataLoader, model, device):
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
            for batch, (x, y) in enumerate(self.data):
                with torch.no_grad():
                    x = x.to(self.device)
                    y = y.to(self.device)
                    prediction = self.model(x)
                    validation_loss = self.loss(prediction, y)
                    validation_losses.append(validation_loss.item())
                    
                    # predicted class
                    _, predicted = torch.max(prediction, 1)

                    total += y.size(0)
                    correct += (predicted == y).sum().item()

                    if batch % 5 == 0:
                        print(f"Validation-> Epoch {epoch} -> Batch {batch}: {validation_loss.item():.4f}")
            
            epoch_acc = 100.0 * correct / total if total > 0 else 0.0
            average_epoch_validation_loss = sum(validation_losses) / len(validation_losses) if validation_losses else 0.0
            
            print(f"Average Epoch Validation Loss {epoch} -> {average_epoch_validation_loss:.4f}")
            print(f"Validation-> Epoch {epoch}: {epoch_acc:.2f}%")
            return average_epoch_validation_loss, epoch_acc

        
        except Exception as e:
            print(f"Error in Validation Loop Epoch {epoch} and Batch No {batch} due to {e}")
            return None

# ===============================
# Example usage
# ===============================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test dataset
    test_dataset = DrowsinessDataset(data_dir="datas/processed/test", train=False)
    test_loader = get_dataloader(test_dataset, batch_size=16, shuffle=False)
    
    # Load trained model
    model = DrowsinessCNN(num_classes=2).to(device)
    model_path = "artifacts/models/drowsiness_cnn.pth"
    model.load_state_dict(torch.load(model_path))
    
    # Initialize evaluator
    evaluator = Evaluator(batch_size=16, data=test_loader, model=model, device=device)
    
    # Run evaluation
    evaluator.start_evaluation_loop(epoch=1)
