# Training Results Summary

## ✅ Training Completed Successfully!

**Date**: February 14, 2026  
**Best Validation Accuracy**: **87.25%** (Epoch 10)  
**Model Location**: `/content/Drowsiness_Detection/artifacts/models/drowsiness_cnn_best.pth`

## 📊 Training Progress

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss | Best Model |
|-------|-----------|---------|------------|----------|------------|
| 1     | 51.12%    | 50.25%  | 0.7821     | 0.6930   | ✅         |
| 2     | 52.56%    | 54.88%  | 0.6922     | 0.6900   | ✅         |
| 3     | 58.44%    | 72.00%  | 0.6708     | 0.5816   | ✅         |
| 4     | 78.94%    | 80.00%  | 0.4525     | 0.4373   | ✅         |
| 5     | 82.09%    | 76.25%  | 0.4169     | 0.4943   |            |
| 6     | 85.34%    | 79.75%  | 0.3589     | 0.4342   |            |
| 7     | 85.75%    | 84.50%  | 0.3550     | 0.3433   | ✅         |
| 8     | 86.53%    | 85.12%  | 0.3271     | 0.3477   | ✅         |
| 9     | 86.62%    | 84.12%  | 0.3259     | 0.3574   |            |
| 10    | 88.16%    | 87.25%  | 0.3047     | 0.3301   | ✅         |

## 🎯 Final Performance (Epoch 10)

### Overall Metrics
- **Training Accuracy**: 88.16%
- **Validation Accuracy**: 87.25%
- **Training Loss**: 0.3047
- **Validation Loss**: 0.3301

### Per-Class Performance
- **Class 0 (closed_1)**: 
  - Training: 93.12% (1490/1600)
  - Validation: 99.50% (398/400) ✅
  
- **Class 1 (opened_1)**:
  - Training: 83.19% (1331/1600)
  - Validation: 75.00% (300/400) ⚠️

## ⚠️ Class Imbalance Issue

**Problem Identified**: The model shows better performance on Class 0 (closed_1) than Class 1 (opened_1):
- Class 0 validation accuracy: **99.50%** (excellent)
- Class 1 validation accuracy: **75.00%** (needs improvement)

**Gap**: 24.5% difference in validation accuracy between classes.

### Why This Happens
Even though the dataset is balanced (1600 samples per class), the model learned to predict "closed" more confidently. This could be due to:
1. Feature differences between classes
2. Data quality/variation differences
3. Model architecture limitations

### Recommendations

1. **Collect More Data for Class 1**: Add more diverse "opened" eye images
2. **Data Augmentation**: Apply more aggressive augmentation to Class 1
3. **Adjust Class Weights**: Manually increase weight for Class 1
4. **Architecture Changes**: Consider deeper network or different architecture
5. **Training Longer**: Train for more epochs with learning rate scheduling

## 📁 Saved Models

1. **Best Model**: `drowsiness_cnn_best.pth`
   - Validation Accuracy: 87.25%
   - Location: `/content/Drowsiness_Detection/artifacts/models/`
   - **Use this for inference!**

2. **Final Model**: `drowsiness_cnn_final.pth`
   - Last epoch model
   - Location: `/content/Drowsiness_Detection/artifacts/models/`

## 🚀 Next Steps

1. **Update app.py**: Model path has been updated to use the new best model
2. **Test the Model**: Run your app and test with various images
3. **Monitor Performance**: Check if predictions are balanced
4. **If Still Biased**: Consider the recommendations above

## 📈 Improvement Over Previous Model

- **Previous Issue**: Model always predicted "Closed"
- **Current Performance**: 87.25% overall accuracy
- **Class 0**: 99.50% accuracy (excellent)
- **Class 1**: 75.00% accuracy (needs improvement but functional)

The model should now predict both classes, though it may still favor "closed" predictions. This is a significant improvement!

## 🔧 To Use the Model

Update your `app/app.py` MODEL_PATH to:
```python
MODEL_PATH = "/content/Drowsiness_Detection/artifacts/models/drowsiness_cnn_best.pth"
```

Or if running locally:
```python
MODEL_PATH = "/home/moeenuddin/Desktop/Deep_learning/drowsiness_detection/Drowsiness_Detection/artifacts/models/drowsiness_cnn_best.pth"
```

## 📝 Notes

- Class names in training: `closed_1` and `opened_1`
- Model uses 2 classes with balanced dataset (1600 samples each)
- Training used class weights (both set to 1.0 since dataset is balanced)
- Learning rate scheduler was active but didn't trigger (no LR reduction needed)
