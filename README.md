# Drowsiness Detection Project

A deep learning-based drowsiness detection system using CNN that can predict drowsiness from images or live webcam feed.

## Prerequisites

- Python 3.12+
- Virtual environment (recommended)
- Trained model file in `artifacts/models/`

## Installation

### 1. Activate Virtual Environment

If you're using a virtual environment (recommended):
```bash
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

Using `uv` (if available):
```bash
uv sync
```

Or using `pip`:
```bash
pip install -r requirements.txt
# Or install from pyproject.toml dependencies
pip install fastapi uvicorn torch torchvision pillow opencv-python python-multipart matplotlib seaborn scikit-learn wandb
```

## Running the Application

### Method 1: Run as Python Module (Recommended)
```bash
cd /home/moeenuddin/Desktop/Deep_learning/drowsiness_detection/Drowsiness_Detection
python3 -m app.app
```

### Method 2: Run with uvicorn directly
```bash
cd /home/moeenuddin/Desktop/Deep_learning/drowsiness_detection/Drowsiness_Detection
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables auto-reload when you make code changes.

### Method 3: Run the app.py file directly
```bash
cd /home/moeenuddin/Desktop/Deep_learning/drowsiness_detection/Drowsiness_Detection
python3 app/app.py
```

## Accessing the Application

Once the server starts, you should see:
```
✅ Model loaded and ready!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Open your web browser and navigate to:
- **http://localhost:8000** or
- **http://127.0.0.1:8000**

## Features

1. **Image Upload Tab**: 
   - Upload an image file
   - Click "Predict" to get drowsiness prediction
   - View confidence scores and probabilities

2. **Webcam Tab**:
   - View live webcam feed
   - Real-time drowsiness predictions overlaid on video

## API Endpoints

- `GET /` - Web UI (HTML interface)
- `POST /predict-image` - Predict drowsiness from uploaded image
- `GET /webcam` - Live webcam stream with predictions

## Troubleshooting

### Model Not Found Error
- Check that the model file exists at: `artifacts/models/drowsiness_epoch10_train88.16_val87.25.pth`
- The app will automatically search for `.pth` files in the models directory

### CUDA/GPU Issues
- The app automatically falls back to CPU if GPU is not compatible
- You'll see a warning but the app will continue running on CPU

### Import Errors
- Make sure you're in the project root directory
- Ensure all dependencies are installed
- Activate your virtual environment if using one

### To Stop the Server
- Press `Ctrl+C` in the terminal where it's running

## Project Structure

```
Drowsiness_Detection/
├── app/
│   └── app.py              # FastAPI application
├── src/
│   ├── data/               # Data processing modules
│   ├── model/              # CNN model and training
│   └── pipelines/         # Training pipelines
├── artifacts/
│   └── models/             # Trained model checkpoints
└── datas/                  # Dataset files
```
