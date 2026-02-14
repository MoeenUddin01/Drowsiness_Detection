# Training Improvements for Better Model Generalization

## 🎯 Problem
The model was always predicting "Closed" (drowsy) class, indicating:
- Class imbalance in the dataset
- Model bias towards one class
- Insufficient generalization

## ✅ Solutions Implemented

### 1. **Class Weight Balancing** ⚖️
- **What**: Automatically calculates class weights based on dataset distribution
- **Why**: Handles imbalanced datasets by giving more weight to underrepresented classes
- **How**: Uses inverse frequency weighting: `weight = total_samples / (num_classes * class_samples)`
- **Location**: `src/data/loader.py` - `calculate_class_weights()` function
- **Usage**: Automatically applied when using `get_dataloaders(calculate_weights=True)`

### 2. **Enhanced Data Augmentation** 🎨
- **What**: More aggressive and diverse data augmentation
- **Why**: Improves model generalization by exposing model to more variations
- **New Augmentations Added**:
  - Random crop (with slight resize first)
  - Random rotation (±10 degrees)
  - Random affine transformations (small translations)
  - Random erasing (10% probability)
  - Enhanced color jitter
- **Location**: `src/data/transforms.py` - `get_train_transform()`

### 3. **Learning Rate Scheduling** 📉
- **What**: Adaptive learning rate that decreases when validation loss plateaus
- **Why**: Prevents overfitting and helps find better minima
- **How**: `ReduceLROnPlateau` scheduler reduces LR by 50% after 3 epochs without improvement
- **Location**: `src/model/train.py` - Trainer class

### 4. **Per-Class Accuracy Tracking** 📊
- **What**: Tracks accuracy for each class separately
- **Why**: Identifies if model is biased towards one class
- **Output**: Shows accuracy for Class 0 and Class 1 separately during training
- **Location**: Both `train.py` and `evaluation.py`

### 5. **Early Stopping** 🛑
- **What**: Stops training if validation accuracy doesn't improve
- **Why**: Prevents overfitting and saves training time
- **How**: Stops after 5 epochs without improvement
- **Location**: `src/pipelines/model_training.py`

### 6. **Best Model Saving** 💾
- **What**: Saves the model with best validation accuracy separately
- **Why**: Ensures you always have the best performing model
- **Files**: 
  - `drowsiness_cnn.pth` - Latest checkpoint
  - `drowsiness_cnn_best.pth` - Best model based on validation accuracy

### 7. **Gradient Clipping** ✂️
- **What**: Limits gradient values to prevent exploding gradients
- **Why**: Stabilizes training, especially with deeper networks
- **How**: Clips gradients to max norm of 1.0
- **Location**: `src/model/train.py` - `train_one_epoch()`

### 8. **Weight Decay (L2 Regularization)** 🛡️
- **What**: Adds L2 penalty to model weights
- **Why**: Prevents overfitting by penalizing large weights
- **How**: `weight_decay=1e-4` in Adam optimizer

## 📝 How to Train

### Option 1: Use the Pipeline Script (Recommended)
```bash
python src/pipelines/model_training.py
```

### Option 2: Use the Simple Training Script
```bash
python src/model/train.py
```

## 🔍 What to Monitor

During training, watch for:

1. **Per-Class Accuracies**: Both classes should have similar accuracies
   ```
   Class 0 Accuracy: 85.23% (1234/1447)
   Class 1 Accuracy: 87.45% (1456/1665)
   ```

2. **Validation vs Training**: Validation accuracy should be close to training accuracy
   - If training >> validation: Model is overfitting
   - If both are low: Model needs more training or better architecture

3. **Learning Rate**: Should decrease over time if scheduler is working
   ```
   Learning Rate: 0.001000 -> 0.000500 -> 0.000250
   ```

4. **Early Stopping**: Should trigger if model stops improving

## 🎯 Expected Results

After these improvements, you should see:
- ✅ Balanced predictions for both classes
- ✅ Similar accuracy for both "Closed" and "Opened" classes
- ✅ Better generalization on new images
- ✅ More stable training process

## 📊 Dataset Balance Check

Before training, the script will show:
```
📈 Calculating class weights for balanced training:
  Class 0 (Closed): 5000 samples, weight: 1.2000
  Class 1 (Opened): 3000 samples, weight: 2.0000
```

If weights are very different (e.g., 1.0 vs 5.0), consider:
- Collecting more data for the underrepresented class
- Using data augmentation more aggressively
- Using techniques like SMOTE or oversampling

## 🚀 Next Steps

1. **Check your dataset balance**: Make sure you have roughly equal samples per class
2. **Run training**: Use the improved training script
3. **Monitor metrics**: Watch per-class accuracies during training
4. **Test the model**: Use the updated app.py to test predictions
5. **Iterate**: If still biased, collect more data or adjust class weights manually

## 🔧 Manual Class Weight Adjustment

If automatic weights don't work, you can set them manually:

```python
# In your training script
class_weights = torch.tensor([2.0, 1.0])  # Give more weight to class 0
trainer = Trainer(..., class_weights=class_weights)
```

## 📚 Additional Resources

- Class imbalance: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Learning rate scheduling: https://pytorch.org/docs/stable/optim.html
- Data augmentation: https://pytorch.org/vision/stable/transforms.html
