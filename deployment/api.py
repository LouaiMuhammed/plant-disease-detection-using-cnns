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

# Initialize FastAPI
app = FastAPI(title="Plant Disease API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cpu')
CHECKPOINT_CANDIDATES = [
    ROOT_DIR / 'models' / 'MobileNet_targeted_hard_finetune.pth',
    ROOT_DIR / 'models' / 'MobileNet_Finetuned_added_data.pth',
    ROOT_DIR / 'models' / 'MobileNet_production.pth',
]
CALIBRATION_PATH = ROOT_DIR / 'models' / 'production_calibration.json'
TREATMENTS_PATH = ROOT_DIR / 'assets' / 'treatments.json'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.502275, 0.528588, 0.474627], std=[0.251866, 0.251658, 0.335502]),
])

CONF_THRESHOLD = 0.40
MARGIN_THRESHOLD = 0.10
FAMILY_GAP_THRESHOLD = 0.30
CLASS_THRESHOLDS = {
    'mango_sooty_mould': {'conf': 0.50, 'margin': 0.15},
}
PAIR_MARGIN_RULES = {
    frozenset(('mango_powdery_mildew', 'mango_sooty_mould')): 0.30,
}

with TREATMENTS_PATH.open('r', encoding='utf-8') as f:
    TREATMENTS = json.load(f)


def _resolve_checkpoint_path():
    for candidate in CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'No checkpoint found in: {CHECKPOINT_CANDIDATES}')


CHECKPOINT_PATH = _resolve_checkpoint_path()


def _load_classes_and_model():
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if 'idx_to_class' in ckpt:
        classes = ckpt['idx_to_class']
    elif 'class_to_idx' in ckpt:
        inv = {v: k for k, v in ckpt['class_to_idx'].items()}
        classes = [inv[i] for i in range(len(inv))]
    else:
        raise RuntimeError('Checkpoint missing class mapping')

    model = get_mobilenet_model(len(classes), version='v2', dropout=0.2).to(DEVICE)
    state = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, classes


def _load_temperature():
    if not CALIBRATION_PATH.exists():
        return 1.0
    with CALIBRATION_PATH.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return float(data.get('temperature', 1.0))


def _display_name(name: str) -> str:
    return name.replace('_', ' ').title()


def _segment_image(image: Image.Image) -> Image.Image:
    image = image.convert('RGB')
    try:
        segmented = remove(image)
        if segmented.mode != 'RGBA':
            return segmented.convert('RGB')

        canvas = Image.new('RGB', segmented.size, (0, 0, 0))
        canvas.paste(segmented, mask=segmented.getchannel('A'))
        return canvas
    except Exception:
        return image


print('Loading production model...')
model, CLASSES = _load_classes_and_model()
TEMPERATURE = _load_temperature()
print(f'Loaded {len(CLASSES)} classes from {CHECKPOINT_PATH.name}')
print(f'Temperature: {TEMPERATURE:.4f}')


@app.get('/')
def home():
    return {
        'status': 'online',
        'model': CHECKPOINT_PATH.name,
        'classes': len(CLASSES),
        'temperature': TEMPERATURE,
    }


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = _segment_image(img)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits / TEMPERATURE, dim=1).squeeze(0)

    top2_probs, top2_idx = torch.topk(probs, k=min(2, len(CLASSES)), dim=0)
    conf = float(top2_probs[0].item())
    idx = int(top2_idx[0].item())
    second_conf = float(top2_probs[1].item()) if len(top2_probs) > 1 else 0.0
    second_idx = int(top2_idx[1].item()) if len(top2_idx) > 1 else idx
    margin = conf - second_conf

    pred_class = CLASSES[idx]
    second_class = CLASSES[second_idx]

    citrus_idx = [i for i, n in enumerate(CLASSES) if n.startswith('citrus_')]
    mango_idx = [i for i, n in enumerate(CLASSES) if n.startswith('mango_')]
    citrus_prob = float(probs[citrus_idx].sum().item()) if citrus_idx else 0.0
    mango_prob = float(probs[mango_idx].sum().item()) if mango_idx else 0.0
    family_gap = abs(citrus_prob - mango_prob)

    conf_th = CONF_THRESHOLD
    margin_th = MARGIN_THRESHOLD
    if pred_class in CLASS_THRESHOLDS:
        conf_th = CLASS_THRESHOLDS[pred_class]['conf']
        margin_th = CLASS_THRESHOLDS[pred_class]['margin']

    uncertain = (conf < conf_th) or (margin < margin_th)
    reason = 'accepted'

    if family_gap < FAMILY_GAP_THRESHOLD:
        uncertain = True
        reason = f'family_rule: gap<{FAMILY_GAP_THRESHOLD:.2f}'

    pair_key = frozenset((pred_class, second_class))
    if pair_key in PAIR_MARGIN_RULES:
        required_margin = PAIR_MARGIN_RULES[pair_key]
        if margin < required_margin:
            uncertain = True
            reason = f'pair_rule: margin<{required_margin:.2f}'
        elif not uncertain:
            reason = 'accepted_pair_rule_passed'
    elif uncertain:
        reason = 'low_conf_or_margin'

    return {
        'predicted_class': _display_name(pred_class),
        'confidence': round(conf, 4),
        'treatment': TREATMENTS.get(pred_class, 'No treatment available.'),
    }


@app.get('/classes')
def get_classes():
    return {
        'classes': CLASSES,
        'total': len(CLASSES),
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
