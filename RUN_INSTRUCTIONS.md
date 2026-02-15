# How to Run the Drowsiness Detection App

## Prerequisites
Make sure you have Python 3.12+ installed and all dependencies.

## Step 1: Activate Virtual Environment (if using one)

```bash
cd /home/moeenuddin/Desktop/Deep_learning/drowsiness_detection/Drowsiness_Detection
source .venv/bin/activate
```

## Step 2: Install Dependencies (if not already installed)

If you're using `uv` (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install fastapi uvicorn torch torchvision pillow opencv-python python-multipart matplotlib seaborn scikit-learn wandb
```

## Step 3: Verify Model File Exists

Make sure your model checkpoint exists at:
```
artifacts/models/drowsiness_epoch10_train88.16_val87.25.pth
```

The app will automatically search for `.pth` files if the exact filename doesn't match.

## Step 4: Run the Application

### Method 1: Run as Python Module (Recommended)
```bash
python3 -m app.app
```

### Method 2: Run with uvicorn command
```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables auto-reload when you make code changes.

### Method 3: Run directly
```bash
python3 app/app.py
```

## Step 5: Access the Application

Once the server starts, you should see:
```
✅ Model loaded and ready!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Open your web browser and go to:
- **http://localhost:8000** or
- **http://127.0.0.1:8000**

## Step 6: Use the Application

1. **Image Upload Tab**: 
   - Click or drag & drop an image
   - Click "Predict" button
   - See the prediction result with confidence scores

2. **Webcam Tab**:
   - View live webcam feed with real-time predictions
   - Status overlay shows "😊 Awake" or "😴 Drowsy"

## Troubleshooting

### If you see "Model not found" error:
- Check that the model file exists in `artifacts/models/`
- The app will automatically search for any `.pth` file in that directory
- Verify the model file path in `app/app.py` (around line 44)

### If you see import errors:
- Make sure you're in the project root directory
- Install missing dependencies: `pip install <package-name>`
- Activate your virtual environment

### If the model always predicts the same class:
- Check the console output for DEBUG messages showing probabilities
- The debug output will help identify if it's a model bias issue or preprocessing problem

### CUDA/GPU Compatibility Issues:
- The app automatically detects incompatible GPUs and falls back to CPU
- You'll see warnings but the app will continue running
- For Quadro P600 (CUDA 6.1), you'll need to use CPU or install a compatible PyTorch version

### To stop the server:
- Press `Ctrl+C` in the terminal where it's running

## API Endpoints

- `GET /` - Web UI (HTML interface)
- `POST /predict-image` - Predict from uploaded image (JSON response)
- `GET /webcam` - Live webcam stream with predictions (MJPEG stream)

## Example API Usage

### Predict from image (using curl):
```bash
curl -X POST "http://localhost:8000/predict-image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"
```
