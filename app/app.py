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
    <title>GuardAI | Drowsiness Monitor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #e2e8f0; }
        .glass-panel { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); }
        .sidebar-link { transition: all 0.2s; border-left: 3px solid transparent; }
        .sidebar-link.active { background: rgba(59, 130, 246, 0.1); border-left-color: #3b82f6; color: #60a5fa; }
        .drag-over { border-color: #3b82f6 !important; background: rgba(59, 130, 246, 0.1) !important; }
        .loader { border-top-color: #3b82f6; animation: spinner 1.5s linear infinite; }
        @keyframes spinner { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body class="h-screen flex overflow-hidden">

    <aside class="w-64 glass-panel flex flex-col z-10">
        <div class="p-6 flex items-center gap-3 border-b border-gray-700">
            <div class="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <i class="fas fa-eye text-white text-sm"></i>
            </div>
            <h1 class="text-xl font-bold tracking-tight text-white">GuardAI</h1>
        </div>
        <nav class="flex-1 py-6">
            <button onclick="switchTab('upload')" id="btn-upload" class="sidebar-link active w-full text-left px-6 py-3 flex items-center gap-3 text-gray-400">
                <i class="fas fa-file-image w-5"></i><span>Image Analysis</span>
            </button>
            <button onclick="switchTab('webcam')" id="btn-webcam" class="sidebar-link w-full text-left px-6 py-3 flex items-center gap-3 text-gray-400">
                <i class="fas fa-video w-5"></i><span>Live Monitor</span>
            </button>
        </nav>
    </aside>

    <main class="flex-1 relative overflow-y-auto">
        <div class="max-w-6xl mx-auto p-8">
            
            <header class="mb-8">
                <h2 class="text-3xl font-bold text-white mb-2" id="page-title">Image Analysis</h2>
                <p class="text-gray-400">Drag images below to analyze drowsiness levels.</p>
            </header>

            <div id="view-upload" class="space-y-6">
                <div id="drop-zone" class="glass-panel rounded-2xl p-1 transition-all">
                    <label for="fileInput" class="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-gray-600 rounded-xl cursor-pointer hover:border-blue-500 transition-colors group">
                        <div class="flex flex-col items-center justify-center pt-5 pb-6">
                            <i class="fas fa-cloud-upload-alt text-3xl text-gray-400 mb-3 group-hover:text-blue-400"></i>
                            <p class="text-lg text-gray-300 font-medium">Drag & Drop or Click to Upload</p>
                        </div>
                        <input id="fileInput" type="file" class="hidden" accept="image/*" onchange="processFile(this.files[0])" />
                    </label>
                </div>

                <div id="results-container" class="hidden grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-in">
                    
                    <div class="glass-panel rounded-2xl p-4">
                        <h3 class="text-xs font-bold text-gray-500 uppercase tracking-widest mb-4">Source Image</h3>
                        <img id="preview-img" src="" class="w-full h-64 object-cover rounded-lg border border-gray-700 shadow-2xl">
                    </div>

                    <div class="glass-panel rounded-2xl p-6 flex flex-col justify-between">
                        <div>
                            <h3 class="text-xs font-bold text-gray-500 uppercase tracking-widest mb-6">AI Analysis</h3>
                            <div id="loading-state" class="hidden flex flex-col items-center py-10">
                                <div class="loader rounded-full border-4 border-gray-700 h-10 w-10 mb-4"></div>
                                <p class="text-blue-400">Analyzing patterns...</p>
                            </div>

                            <div id="data-display" class="text-center">
                                <div id="status-icon" class="text-6xl mb-4"></div>
                                <div id="prediction-text" class="text-4xl font-black text-white mb-8"></div>
                                
                                <div class="space-y-6 text-left">
                                    <div>
                                        <div class="flex justify-between text-sm mb-2"><span class="text-gray-400">Awake Confidence</span><span id="prob-open">0%</span></div>
                                        <div class="w-full bg-gray-800 rounded-full h-3"><div id="bar-open" class="bg-green-500 h-3 rounded-full transition-all duration-1000" style="width: 0%"></div></div>
                                    </div>
                                    <div>
                                        <div class="flex justify-between text-sm mb-2"><span class="text-gray-400">Drowsy Confidence</span><span id="prob-closed">0%</span></div>
                                        <div class="w-full bg-gray-800 rounded-full h-3"><div id="bar-closed" class="bg-red-500 h-3 rounded-full transition-all duration-1000" style="width: 0%"></div></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div id="view-webcam" class="hidden">
                <div class="glass-panel rounded-2xl p-1"><img id="webcam-feed" src="" class="w-full rounded-xl bg-black min-h-[480px]"></div>
            </div>
        </div>
    </main>

    <script>
        const dropZone = document.getElementById('drop-zone');

        // Drag and Drop Logic
        ['dragenter', 'dragover'].forEach(name => {
            dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); }, false);
        });
        ['dragleave', 'drop'].forEach(name => {
            dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.classList.remove('drag-over'); }, false);
        });
        dropZone.addEventListener('drop', (e) => { processFile(e.dataTransfer.files[0]); }, false);

        function processFile(file) {
            if (!file) return;

            // Show Results Container
            document.getElementById('results-container').classList.remove('hidden');
            document.getElementById('data-display').classList.add('hidden');
            document.getElementById('loading-state').classList.remove('hidden');

            // Set Preview Image
            const reader = new FileReader();
            reader.onload = (e) => { document.getElementById('preview-img').src = e.target.result; };
            reader.readAsDataURL(file);

            // Send to FastAPI
            const formData = new FormData();
            formData.append('file', file);
            fetch('/predict-image', { method: 'POST', body: formData })
                .then(res => res.json())
                .then(data => {
                    document.getElementById('loading-state').classList.add('hidden');
                    document.getElementById('data-display').classList.remove('hidden');
                    updateUI(data);
                });
        }

        function updateUI(data) {
            const isAwake = data.class_index === 1;
            const statusText = document.getElementById('prediction-text');
            statusText.innerText = isAwake ? "AWAKE" : "DROWSY";
            statusText.className = isAwake ? "text-4xl font-black text-green-400 mb-8" : "text-4xl font-black text-red-500 mb-8";
            document.getElementById('status-icon').innerHTML = isAwake ? '<i class="fas fa-check-circle text-green-400"></i>' : '<i class="fas fa-exclamation-triangle text-red-500"></i>';

            const pOpen = (data.probabilities.opened * 100).toFixed(1);
            const pClosed = (data.probabilities.closed * 100).toFixed(1);
            document.getElementById('prob-open').innerText = pOpen + "%";
            document.getElementById('bar-open').style.width = pOpen + "%";
            document.getElementById('prob-closed').innerText = pClosed + "%";
            document.getElementById('bar-closed').style.width = pClosed + "%";
        }

        function switchTab(tab) {
            document.querySelectorAll('.sidebar-link').forEach(el => el.classList.remove('active'));
            document.getElementById('btn-' + tab).classList.add('active');
            document.getElementById('view-upload').classList.toggle('hidden', tab !== 'upload');
            document.getElementById('view-webcam').classList.toggle('hidden', tab !== 'webcam');
            document.getElementById('webcam-feed').src = tab === 'webcam' ? "/webcam" : "";
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