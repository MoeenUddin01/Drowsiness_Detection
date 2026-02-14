# How to Run the Drowsiness Detection App

## Prerequisites
Make sure you have Python 3.10+ installed and all dependencies.

## Step 1: Install Dependencies (if not already installed)

If you're using `uv` (recommended):
```bash
cd /home/moeenuddin/Desktop/Deep_learning/drowsiness_detection/Drowsiness_Detection
uv sync
```

Or using pip:
```bash
pip install fastapi uvicorn torch torchvision pillow opencv-python python-multipart
```

## Step 2: Verify Model File Exists

Make sure your model checkpoint exists at:
```
artifacts/models/drowsiness_epoch10_train97.66_val94.00.pth
```

## Step 3: Run the Application

### Method 1: Run directly with Python
```bash
cd /home/moeenuddin/Desktop/Deep_learning/drowsiness_detection/Drowsiness_Detection
python3 app/app.py
```

### Method 2: Run with uvicorn command
```bash
cd /home/moeenuddin/Desktop/Deep_learning/drowsiness_detection/Drowsiness_Detection
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables auto-reload when you make code changes.

## Step 4: Access the Application

Once the server starts, you should see:
```
✅ Model loaded and ready!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Open your web browser and go to:
- **http://localhost:8000** or
- **http://127.0.0.1:8000**

## Step 5: Use the Application

1. **Image Upload Tab**: 
   - Click or drag & drop an image
   - Click "Predict" button
   - See the prediction result

2. **Webcam Tab**:
   - View live webcam feed with real-time predictions

## Troubleshooting

### If you see "Model not found" error:
- Check that the model file path in `app/app.py` (line 20) is correct
- Verify the model file exists at that location

### If you see import errors:
- Make sure you're in the project root directory
- Install missing dependencies: `pip install <package-name>`

### If the model always predicts the same class:
- Check the console output for DEBUG messages showing probabilities
- The debug output will help identify if it's a model bias issue or preprocessing problem

### To stop the server:
- Press `Ctrl+C` in the terminal where it's running

## API Endpoints

- `GET /` - Web UI
- `POST /predict-image` - Predict from uploaded image
- `GET /webcam` - Live webcam stream with predictions
