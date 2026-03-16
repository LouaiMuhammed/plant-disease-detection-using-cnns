"""
REST API for Plant Disease Classification
Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""
import io
import json
import sys
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from rembg import remove

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models import get_mobilenet_model

app = FastAPI(title="Plant Disease API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cpu')
CHECKPOINT_PATH = ROOT_DIR / 'models' / 'MobileNet_segmented_data.pth'
TREATMENTS_PATH = ROOT_DIR / 'assets' / 'treatments.json'

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


def _load_model_and_classes():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f'Checkpoint not found: {CHECKPOINT_PATH}')

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
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


print('Loading model...')
MODEL, CLASSES = _load_model_and_classes()
print(f'Ready — {len(CLASSES)} classes from {CHECKPOINT_PATH.name}')


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

    return {
        'predicted_class': _display_name(pred_class),
        'confidence': round(conf, 4),
        'treatment': TREATMENTS.get(pred_class, 'No treatment available.'),
    }


@app.get('/classes')
def get_classes():
    return {'classes': CLASSES, 'total': len(CLASSES)}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)