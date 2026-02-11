# app/app.py

import io
import cv2
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from src.model.cnn import CNN  # import your trained CNN

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/home/moeenuddin/Desktop/Deep_learning/drowsiness_detection/Drowsiness_Detection/artifacts/models/drowsiness_epoch5_train94.50_val93.12.pth"
IMAGE_SIZE = 128
MODEL_NUM_CLASSES = 13  # Model trained with 13 classes

# Class names: Your model outputs probabilities for 13 classes
# We interpret: majority of classes (7-12) = Closed/Drowsy, minority (0-6) = Open/Awake
CLASS_NAMES_2 = ["Awake", "Drowsy"]

# Class mapping: Sum probabilities
# Classes 0-6: Awake (open eyes)
# Classes 7-12: Drowsy (closed eyes)
AWAKE_CLASSES = list(range(7))
DROWSY_CLASSES = list(range(7, 13))

# -----------------------------
# IMAGE TRANSFORMS (matching training transforms)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Simple transform for webcam (without normalization for now)
transform_simple = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# # Alternative: If model classes represent a spectrum, use confidence threshold
def predict_drowsiness(probabilities):
    """
    Predict Awake (0) or Drowsy (1) by summing class probabilities
    Args:
        probabilities: torch tensor of shape [1, 13]
    Returns:
        class_idx (0 or 1), confidence score
    """
    awake_prob = probabilities[0, AWAKE_CLASSES].sum().item()
    drowsy_prob = probabilities[0, DROWSY_CLASSES].sum().item()
    
    if drowsy_prob > awake_prob:
        return 1, drowsy_prob / (awake_prob + drowsy_prob)
    else:
        return 0, awake_prob / (awake_prob + drowsy_prob)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = CNN(num_classes=MODEL_NUM_CLASSES).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()
print("✅ Model loaded and ready!")

# -----------------------------
# IMAGE TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

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
            resultDiv.innerHTML = `
                <div class="prediction ${predClass}">
                    ${predClass === 'awake' ? '😊 Awake' : '😴 Drowsy'}
                </div>
                <div class="confidence">
                    <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%<br>
                    <strong>Model Output:</strong> Class ${data.model_output_class}
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
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Predict using summed probabilities
        label_idx, confidence = predict_drowsiness(probabilities)

        return JSONResponse({
            "prediction": CLASS_NAMES_2[label_idx],
            "class_index": label_idx,
            "confidence": round(float(confidence), 4),
            "status": "Drowsy" if label_idx == 1 else "Awake"
        })
    except Exception as e:
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
        pil_image = Image.fromarray(rgb_frame)
        input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Predict using summed probabilities
        label_idx, _ = predict_drowsiness(probabilities)
        label = CLASS_NAMES_2[label_idx]
        color = (0, 255, 0) if label == "Awake" else (0, 0, 255)
        cv2.putText(frame, f"Status: {label}", (20, 40),
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
