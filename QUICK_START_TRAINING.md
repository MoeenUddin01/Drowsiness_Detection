# Quick Start: Training Your Model

## ✅ Yes, you can run the training script!

The script `src/pipelines/model_training.py` is now fixed and ready to run on your local machine.

## 📋 Prerequisites

### 1. **Prepare Your Dataset**

Your dataset should be organized like this:
```
datas/
  processed/
    train/
      Closed/    (images of closed eyes)
      Opened/    (images of open eyes)
    test/
      Closed/    (images of closed eyes)
      Opened/    (images of open eyes)
```

**If you don't have this structure yet:**

1. **Option A: Use the train-test split script**
   ```bash
   python src/data/train_test_split.py
   ```
   This will split your `datas/raw/` folder into train/test.

2. **Option B: Organize manually**
   - Create `datas/processed/train/Closed/` and `datas/processed/train/Opened/`
   - Create `datas/processed/test/Closed/` and `datas/processed/test/Opened/`
   - Put your images in the appropriate folders

### 2. **Install Dependencies** (if not already installed)
```bash
pip install torch torchvision pillow opencv-python
# Optional: for W&B logging
pip install wandb
```

## 🚀 How to Run

### Simple Command:
```bash
cd /home/moeenuddin/Desktop/Deep_learning/drowsiness_detection/Drowsiness_Detection
python3 src/pipelines/model_training.py
```

## 📊 What Will Happen

1. **Dataset Check**: Script will detect your classes automatically
2. **Class Weights**: Will calculate weights to balance imbalanced classes
3. **Training Starts**: You'll see progress for each epoch
4. **Validation**: After each epoch, model is evaluated on test set
5. **Best Model Saved**: Best model saved to `artifacts/models/drowsiness_cnn_best.pth`

## 📝 Expected Output

```
✅ Auto-detected 2 classes: ['Closed', 'Opened']
📊 Train samples: 5000, Test samples: 1000

📈 Calculating class weights for balanced training:
  Class 0 (Closed): 3000 samples, weight: 1.3333
  Class 1 (Opened): 2000 samples, weight: 2.0000

Using device: cuda
Model initialized with 2 classes

[Epoch 1] Batch 0: Loss = 0.6931, LR = 0.001000
...
[Epoch 1] Overall Accuracy: 75.23%
  Class 0 Accuracy: 74.50% (2235/3000)
  Class 1 Accuracy: 76.33% (1527/2000)

[Epoch 1] Average Validation Loss: 0.5234
[Epoch 1] Overall Validation Accuracy: 78.50%
  Class 0 Val Accuracy: 77.20% (231/300)
  Class 1 Val Accuracy: 80.00% (160/200)
```

## ⚠️ Common Issues

### Issue 1: "No such file or directory: datas/processed/train"
**Solution**: Prepare your dataset first (see Prerequisites above)

### Issue 2: "CUDA out of memory"
**Solution**: Reduce batch size in the script:
```python
BATCH_SIZE = 16  # Instead of 32
```

### Issue 3: "W&B login failed"
**Solution**: This is optional! Training will continue without W&B. To use W&B:
```bash
export WANDB_API_KEY=your_key_here
```

### Issue 4: Empty dataset
**Solution**: Make sure you have images in:
- `datas/processed/train/Closed/`
- `datas/processed/train/Opened/`
- `datas/processed/test/Closed/`
- `datas/processed/test/Opened/`

## 🎯 After Training

1. **Best model location**: `artifacts/models/drowsiness_cnn_best.pth`
2. **Update app.py**: Change the MODEL_PATH in `app/app.py` to point to your new model
3. **Test**: Run your app and test predictions!

## 📈 Monitoring Training

- **Per-class accuracies**: Watch that both classes have similar accuracies
- **Validation vs Training**: Validation should be close to training (not much lower)
- **Learning rate**: Should decrease if validation loss plateaus
- **Early stopping**: Will stop if no improvement for 5 epochs

## 💡 Tips

1. **Start with fewer epochs** to test: Change `EPOCHS = 5` in the script
2. **Monitor GPU usage**: `nvidia-smi` (if using CUDA)
3. **Check dataset balance**: The script will show class distribution
4. **Save checkpoints**: Best model is saved automatically

## 🆘 Need Help?

If training doesn't start:
1. Check that your dataset is in the right place
2. Verify Python can import all modules: `python3 -c "from src.data.loader import get_dataloaders"`
3. Check error messages - they usually tell you what's missing

Good luck with training! 🚀
