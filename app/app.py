import io
import cv2
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.cnn import CNN  # import your trained CNN

# -----------------------------
# CONFIG
# -----------------------------
# Check if CUDA is available and compatible
def get_device():
    """Get the best available device, checking CUDA compatibility"""
    if torch.cuda.is_available():
        try:
            # Try a simple CUDA operation to check compatibility
            test_tensor = torch.tensor([1.0]).cuda()
            _ = test_tensor * 2
            del test_tensor
            torch.cuda.empty_cache()
            print("🖥️  Using device: CUDA (GPU)")
            return "cuda"
        except Exception as e:
            print(f"⚠️  CUDA available but not compatible: {str(e)[:100]}")
            print("   Falling back to CPU")
            return "cpu"
    print("🖥️  Using device: CPU")
    return "cpu"

DEVICE = get_device()
# Model path - auto-detect local or Colab path
import os

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "drowsiness_epoch10_train88.16_val87.25.pth"
COLAB_MODEL_PATH = "/content/Drowsiness_Detection/artifacts/models/drowsiness_cnn_best.pth"

# Use local path if it exists, otherwise try Colab path
if LOCAL_MODEL_PATH.exists():
    MODEL_PATH = str(LOCAL_MODEL_PATH)
elif os.path.exists(COLAB_MODEL_PATH):
    MODEL_PATH = COLAB_MODEL_PATH
else:
    # Fallback: try to find any .pth file in models directory
    models_dir = PROJECT_ROOT / "artifacts" / "models"
    if models_dir.exists():
        pth_files = list(models_dir.glob("*.pth"))
        if pth_files:
            MODEL_PATH = str(pth_files[0])  # Use the first .pth file found
            print(f"⚠️  Using model: {MODEL_PATH}")
        else:
            raise FileNotFoundError(f"No model files found in {models_dir}")
    else:
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
IMAGE_SIZE = 128
MODEL_NUM_CLASSES = 2  # Model trained with 2 classes: closed_1 (Drowsy) and opened_1 (Awake)

# Class names mapping (based on your training data)
# Class 0: "closed_1" (eyes closed = Drowsy)
# Class 1: "opened_1" (eyes open = Awake)
CLASS_NAMES = ["closed_1", "opened_1"]
CLASS_DISPLAY_2 = ["😴 Drowsy", "😊 Awake"]

# -----------------------------
# IMAGE TRANSFORMS (matching training transforms - using test transform)
# -----------------------------
# Import the exact same transform used during testing
from src.data.transforms import get_test_transform

transform = get_test_transform(image_size=(IMAGE_SIZE, IMAGE_SIZE))

# Simple transform for webcam (without normalization for now)
transform_simple = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# # Alternative: If model classes represent a spectrum, use confidence threshold
def predict_drowsiness(probabilities):
    """
    Predict Closed (0 / Drowsy) or Opened (1 / Awake) from 2-class model
    Args:
        probabilities: torch tensor of shape [1, 2]
    Returns:
        class_idx (0 or 1), confidence score
    """
    probs = probabilities[0].cpu()  # Move to CPU for processing
    confidence = probs.max().item()
    class_idx = probs.argmax().item()
    
    # Debug: Print probabilities to help diagnose issues
    print(f"DEBUG - Probabilities: Closed={probs[0].item():.4f}, Opened={probs[1].item():.4f}")
    print(f"DEBUG - Predicted class: {class_idx} ({'Closed' if class_idx == 0 else 'Opened'}), Confidence: {confidence:.4f}")
    
    return class_idx, confidence

# -----------------------------
# LOAD MODEL
# -----------------------------
# Always load model to CPU first, then move to device (handles incompatible GPUs)
model = CNN(num_classes=MODEL_NUM_CLASSES)
try:
    # Load checkpoint to CPU first to avoid CUDA issues
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Model loaded from checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Model loaded from checkpoint")
    
    # Move model to the selected device after loading
    model = model.to(DEVICE)
    model.eval()
    
    # Verify model with a dummy input (skip if CUDA incompatible)
    try:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
            dummy_output = model(dummy_input)
            dummy_probs = torch.nn.functional.softmax(dummy_output, dim=1)
            print(f"✅ Model verification - Output shape: {dummy_output.shape}, Probabilities: {dummy_probs.cpu().numpy()}")
    except Exception as e:
        if "CUDA" in str(e) or "cuda" in str(e).lower() or "kernel" in str(e).lower():
            print(f"⚠️  CUDA error during verification. Switching to CPU...")
            DEVICE = "cpu"
            model = model.cpu()  # Move model to CPU
            # Retry verification on CPU
            try:
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
                    dummy_output = model(dummy_input)
                    dummy_probs = torch.nn.functional.softmax(dummy_output, dim=1)
                    print(f"✅ Model verification on CPU - Output shape: {dummy_output.shape}")
            except Exception as e2:
                print(f"⚠️  Model verification skipped: {e2}")
        else:
            print(f"⚠️  Model verification skipped: {e}")
    
    print("✅ Model loaded and ready!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(
    title="Drowsiness Detection API",
    description="Predict drowsiness from images or webcam",
    version="1.0.0"
)

# HTML UI
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Drowsiness Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px; }
        .container { background: white; border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); width: 100%; max-width: 600px; padding: 40px; }
        h1 { text-align: center; color: #333; margin-bottom: 10px; font-size: 2.5em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; }
        .tabs { display: flex; gap: 10px; margin-bottom: 30px; border-bottom: 2px solid #eee; }
        .tab-btn { padding: 12px 24px; border: none; background: none; cursor: pointer; font-size: 1em; color: #666; border-bottom: 3px solid transparent; transition: all 0.3s; }
        .tab-btn.active { color: #667eea; border-bottom-color: #667eea; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .upload-area { border: 3px dashed #667eea; border-radius: 10px; padding: 40px; text-align: center; cursor: pointer; transition: all 0.3s; }
        .upload-area:hover { background: #f0f4ff; }
        .upload-area.dragover { background: #e8f0ff; border-color: #764ba2; }
        input[type="file"] { display: none; }
        .btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 12px 30px; border-radius: 8px; cursor: pointer; font-size: 1em; transition: transform 0.2s; }
        .btn:hover { transform: translateY(-2px); }
        .btn:active { transform: translateY(0); }
        .result { margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px; display: none; }
        .result.show { display: block; animation: slideIn 0.3s ease; }
        @keyframes slideIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .prediction { font-size: 1.5em; font-weight: bold; padding: 15px; border-radius: 8px; margin: 10px 0; text-align: center; }
        .awake { background: #d4edda; color: #155724; }
        .drowsy { background: #f8d7da; color: #721c24; }
        .confidence { color: #666; margin-top: 10px; }
        .webcam-container { position: relative; width: 100%; max-width: 100%; }
        video { width: 100%; border-radius: 10px; }
        .webcam-btn { margin-top: 15px; }
        .spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid #f3f3f3; border-top: 3px solid #667eea; border-radius: 50%; animation: spin 1s linear infinite; vertical-align: middle; margin-right: 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .loading { display: none; text-align: center; color: #667eea; margin: 20px 0; }
        .error { background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; margin: 10px 0; display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>😴 Drowsiness Detection</h1>
        <p class="subtitle">AI-powered drowsiness prediction system</p>
        
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('upload')">📤 Image Upload</button>
            <button class="tab-btn" onclick="switchTab('webcam')">📹 Webcam</button>
        </div>

        <!-- Image Upload Tab -->
        <div id="upload" class="tab-content active">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()" ondrop="handleDrop(event)" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)">
                <div style="font-size: 3em; margin-bottom: 10px;">📸</div>
                <p><strong>Click to upload</strong> or drag and drop</p>
                <p style="color: #999; font-size: 0.9em;">PNG, JPG, GIF (max 10MB)</p>
            </div>
            <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
            <button class="btn" style="width: 100%; margin-top: 20px;" onclick="predictImage()">Predict</button>
            <div class="loading" id="uploadLoading"><span class="spinner"></span> Analyzing image...</div>
            <div class="error" id="uploadError"></div>
            <div class="result" id="uploadResult"></div>
        </div>

        <!-- Webcam Tab -->
        <div id="webcam" class="tab-content">
            <div class="webcam-container">
                <img id="webcamStream" src="/webcam" style="width: 100%; border-radius: 10px; background: #000;">
            </div>
            <p style="color: #666; margin-top: 10px; text-align: center; font-size: 0.9em;">Live predictions displayed on video feed</p>
        </div>
    </div>

    <script>
        let selectedFile = null;

        function switchTab(tab) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tab).classList.add('active');
            event.target.classList.add('active');
        }

        function handleDragOver(e) {
            e.preventDefault();
            document.querySelector('.upload-area').classList.add('dragover');
        }

        function handleDragLeave(e) {
            document.querySelector('.upload-area').classList.remove('dragover');
        }

        function handleDrop(e) {
            e.preventDefault();
            document.querySelector('.upload-area').classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                selectedFile = files[0];
                document.querySelector('.upload-area').innerHTML = `<div style="font-size: 2em;">✅</div><p>${selectedFile.name}</p><p style="color: #999; font-size: 0.9em;">Ready to predict</p>`;
            }
        }

        function handleFileSelect(e) {
            selectedFile = e.target.files[0];
            if (selectedFile) {
                document.querySelector('.upload-area').innerHTML = `<div style="font-size: 2em;">✅</div><p>${selectedFile.name}</p><p style="color: #999; font-size: 0.9em;">Ready to predict</p>`;
            }
        }

        async function predictImage() {
            if (!selectedFile) {
                showError('Please select an image first');
                return;
            }

            const formData = new FormData();
            formData.append('file', selectedFile);

            showLoading(true);
            showError('');
            
            try {
                const response = await fetch('/predict-image', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error('Prediction failed');
                
                const data = await response.json();
                displayResult(data);
            } catch (error) {
                showError('Error: ' + error.message);
            } finally {
                showLoading(false);
            }
        }

        function displayResult(data) {
            const predClass = data.prediction.toLowerCase();
            const resultDiv = document.getElementById('uploadResult');
            // Handle both 'opened' and 'opened_1' class names
            const isOpened = predClass === 'opened' || predClass === 'opened_1';
            const emoji = isOpened ? '😊' : '😴';
            const status = isOpened ? 'Awake' : 'Drowsy';
            const styleClass = isOpened ? 'awake' : 'drowsy';
            
            resultDiv.innerHTML = `
                <div class="prediction ${styleClass}">
                    ${emoji} ${status}
                </div>
                <div class="confidence">
                    <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%<br>
                    <strong>Model Class:</strong> ${data.prediction} (Class ${data.class_index})
                </div>
            `;
            resultDiv.classList.add('show');
        }

        function showLoading(show) {
            document.getElementById('uploadLoading').style.display = show ? 'block' : 'none';
        }

        function showError(msg) {
            const errorDiv = document.getElementById('uploadError');
            if (msg) {
                errorDiv.textContent = msg;
                errorDiv.style.display = 'block';
            } else {
                errorDiv.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def root():
    from fastapi.responses import HTMLResponse
    return HTMLResponse(HTML_UI)

# -----------------------------
# IMAGE PREDICTION
# -----------------------------
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Ensure image is properly formatted
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Debug: Check tensor shape and stats
        print(f"DEBUG - Input tensor shape: {input_tensor.shape}")
        print(f"DEBUG - Input tensor range: [{input_tensor.min().item():.4f}, {input_tensor.max().item():.4f}]")
        print(f"DEBUG - Input tensor mean: {input_tensor.mean().item():.4f}")

        with torch.no_grad():
            outputs = model(input_tensor)
            # Debug: Check raw outputs before softmax
            print(f"DEBUG - Raw model outputs: {outputs.cpu().numpy()}")
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Predict using 2-class model
        class_idx, confidence = predict_drowsiness(probabilities)
        
        # Get both probabilities for better debugging
        probs = probabilities[0].cpu()
        closed_prob = probs[0].item()
        opened_prob = probs[1].item()

        return JSONResponse({
            "prediction": CLASS_NAMES[class_idx],
            "class_index": class_idx,
            "confidence": round(float(confidence), 4),
            "display_status": CLASS_DISPLAY_2[class_idx],
            "probabilities": {
                "closed": round(float(closed_prob), 4),
                "opened": round(float(opened_prob), 4)
            }
        })
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"ERROR in predict_image: {error_msg}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# -----------------------------
# WEBCAM STREAM
# -----------------------------
def webcam_stream():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot access webcam")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert to RGB and PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame).convert("RGB")
        input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Predict using 2-class model
        class_idx, _ = predict_drowsiness(probabilities)
        label = CLASS_NAMES[class_idx]
        display_label = CLASS_DISPLAY_2[class_idx]
        color = (0, 255, 0) if class_idx == 1 else (0, 0, 255)  # Green for Opened, Red for Closed
        cv2.putText(frame, f"Status: {display_label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()

# -----------------------------
# WEBCAM ENDPOINT
# -----------------------------
@app.get("/webcam")
def webcam_feed():
    return StreamingResponse(
        webcam_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# -----------------------------
# RUN DIRECTLY
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)