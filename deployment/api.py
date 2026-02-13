"""
REST API for Plant Disease Classification
Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

# Initialize FastAPI
app = FastAPI(title="Plant Disease API", version="1.0")

# Enable CORS (so Flutter can connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Disease classes
CLASSES = [
    'citrus_black_spot', 'citrus_canker', 'citrus_foliage_damage',
    'citrus_greening', 'citrus_healthy', 'citrus_mealybugs', 'citrus_melanose',
    'mango_anthracnose', 'mango_bacterial_canker', 'mango_cutting_weevil',
    'mango_die_back', 'mango_gall_midge', 'mango_healthy',
    'mango_powdery_mildew', 'mango_sooty_mould'
]

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model once at startup
print("Loading model...")
model = torch.jit.load('models/mobilenet_v2_plant_disease.pt', map_location='cpu')
model.eval()
print("✓ Model loaded and ready")


@app.get("/")
def home():
    """Health check endpoint"""
    return {
        "status": "online",
        "model": "loaded",
        "version": "1.0"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict disease from uploaded image
    
    Usage:
        curl -X POST -F "file=@leaf.jpg" http://localhost:8000/predict
    
    Returns:
        {
            "predicted_class": "citrus_canker",
            "confidence": 0.985
        }
    """
    # Read uploaded file
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Preprocess
    img_tensor = transform(img).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1).squeeze()
    
    # Get top prediction
    conf, idx = torch.max(probs, dim=0)
    
    return {
        "predicted_class": CLASSES[idx.item()],
        "confidence": float(conf.item()),
        "all_probabilities": {
            CLASSES[i]: float(probs[i].item()) 
            for i in range(len(CLASSES))
        }
    }


@app.get("/classes")
def get_classes():
    """Get list of all disease classes"""
    return {
        "classes": CLASSES,
        "total": len(CLASSES)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)