"""
REST API for Plant Disease Classification
Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""
import io
import json
import logging
import sys
import time
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from rembg import remove, new_session

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models import get_mobilenet_model

app = FastAPI(title="Plant Disease API", version="1.0")

LOG_PATH = Path(__file__).resolve().parent / "server_requests.log"
logger = logging.getLogger("plant_disease_api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cpu')
CHECKPOINT_PATH = ROOT_DIR / 'deployment' / 'models' / 'mobilenet_v2_plant_disease_segmented.pt'
MODEL_METADATA_PATH = ROOT_DIR / 'deployment' / 'models' / 'model_metadata.json'
TREATMENTS_PATH = ROOT_DIR / 'assets' / 'treatments.json'
DESCRIPTION_PATH = ROOT_DIR / 'assets' / 'diseases_description.json'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

with TREATMENTS_PATH.open('r', encoding='utf-8') as f:
    TREATMENTS = json.load(f)

with DESCRIPTION_PATH.open('r', encoding='utf-8') as f:
    DESCRIPTIONS = json.load(f)


def _load_classes_from_metadata():
    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(f'Model metadata not found: {MODEL_METADATA_PATH}')

    with MODEL_METADATA_PATH.open('r', encoding='utf-8') as f:
        metadata = json.load(f)

    classes = metadata.get('classes')
    if not classes:
        raise RuntimeError('Model metadata missing classes list')
    return classes


def _load_model_and_classes():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f'Checkpoint not found: {CHECKPOINT_PATH}')

    try:
        model = torch.jit.load(str(CHECKPOINT_PATH), map_location=DEVICE)
        classes = _load_classes_from_metadata()
        model.eval()
        return model, classes
    except RuntimeError:
        pass

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

    if not isinstance(ckpt, dict):
        classes = _load_classes_from_metadata()
        ckpt.eval()
        return ckpt, classes

    if 'idx_to_class' in ckpt:
        classes = ckpt['idx_to_class']
    elif 'class_to_idx' in ckpt:
        inv = {v: k for k, v in ckpt['class_to_idx'].items()}
        classes = [inv[i] for i in range(len(inv))]
    else:
        raise RuntimeError('Checkpoint missing class mapping')

    model = get_mobilenet_model(len(classes), version='v2', dropout=0.2).to(DEVICE)
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, classes

#REMBG_SESSION = new_session("u2net")  # or "u2netp" for faster/lighter model

def _segment(image: Image.Image) -> Image.Image:
    try:
        segmented = remove(image.convert('RGB'))
        canvas = Image.new('RGB', segmented.size, (0, 0, 0))
        canvas.paste(segmented, mask=segmented.getchannel('A'))
        return canvas
    except Exception:
        return image


def _display_name(name: str) -> str:
    return name.replace('_', ' ').title()


def _severity_level(confidence: float, prediction: str) -> str:
    conf = confidence
    pred = prediction
    if not pred.endswith("_healthy"):
        if conf >= 0.85:
            return "Severe"
        elif conf >= 0.65:
            return "Moderate"
        else:
            return "Low"
    else:
        return "None"


print('Loading model...')
MODEL, CLASSES = _load_model_and_classes()
print(f'Ready — {len(CLASSES)} classes from {CHECKPOINT_PATH.name}')


@app.middleware("http")
async def log_requests(request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000
    client_host = request.client.host if request.client else "unknown"
    logger.info(
        "%s %s | status=%s | client=%s | duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        client_host,
        duration_ms,
    )
    return response


@app.get('/')
def home():
    return {'status': 'online', 'classes': len(CLASSES)}


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = _segment(img)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(MODEL(img_tensor), dim=1).squeeze(0)

    conf, idx = float(probs.max()), int(probs.argmax())
    pred_class = CLASSES[idx]
    severity_level = _severity_level(confidence=conf, prediction=pred_class)
    return {
        'predicted_class': _display_name(pred_class),
        'confidence': round(conf, 2),
        'disease_description': DESCRIPTIONS.get(pred_class),
        'treatment': TREATMENTS.get(pred_class, 'No treatment available.'),
        'severity_level': severity_level,


    }


@app.get('/classes')
def get_classes():
    return {'classes': CLASSES, 'total': len(CLASSES)}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
