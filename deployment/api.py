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
from typing import Dict, List, Tuple
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
from rembg import remove, new_session

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models import get_mobilenet_model

app = FastAPI(title="Plant Disease API", version="2.0 - With Severity & Progress")

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
CHECKPOINT_PATH = ROOT_DIR / 'deployment' / 'models' / 'densenet121_plant_disease_segmented.pt'
MODEL_METADATA_PATH = ROOT_DIR / 'deployment' / 'models' / 'model_metadata.json'
TREATMENTS_PATH = ROOT_DIR / 'assets' / 'treatments.json'
DESCRIPTION_PATH = ROOT_DIR / 'assets' / 'diseases_description.json'
CALIB_PATH = ROOT_DIR / 'models' / 'densenet121_production_calibration.json'

# === SEVERITY & PROGRESS CONFIGURATION ===
TEMPERATURE = 1.0  # Will be loaded from calibration file if available
SEVERITY_THRESHOLDS = {
    # Disease/Healthy probability ratio thresholds for severity bins
    # ratio > threshold -> next severity level
    'mild': 1.0,      # disease_prob / healthy_prob > 1.0 = mild
    'moderate': 3.0,  # disease_prob / healthy_prob > 3.0 = moderate
    'severe': 8.0,    # disease_prob / healthy_prob > 8.0 = severe
}
PROGRESS_THRESHOLDS = {
    'confidence': 0.40,
    'margin': 0.10,
    'improvement_gate': 15,  # Minimum severity point drop to signal improvement
}
FAMILY_CONSISTENCY_THRESHOLD = 0.30  # Gap between citrus% and mango%

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


def _load_calibration():
    """Load temperature and thresholds from calibration file if available."""
    global TEMPERATURE
    
    if CALIB_PATH.exists():
        try:
            with open(CALIB_PATH, 'r', encoding='utf-8') as f:
                calib = json.load(f)
            TEMPERATURE = float(calib.get('temperature', 1.0))
            logger.info(f"Loaded temperature: {TEMPERATURE:.4f}")
            return calib
        except Exception as e:
            logger.warning(f"Could not load calibration: {e}")
    return None


def _compute_severity_score(probs: torch.Tensor, classes: list, pred_idx: int) -> Dict:
    """
    Compute severity score using disease/healthy probability ratio.
    
    Finds corresponding healthy class and computes log-ratio.
    Returns dict with score (0-100), category (mild/moderate/severe), and confidence.
    """
    pred_class = classes[pred_idx]
    
    # Find healthy class for same plant type
    if pred_class.endswith('_healthy'):
        return {
            'score': 0,
            'category': 'Healthy',
            'ratio': None,
            'disease_prob': 0.0,
            'healthy_prob': float(probs[pred_idx].item()),
            'confidence': 'High',
        }
    
    # Determine plant family
    if pred_class.startswith('citrus_'):
        healthy_candidates = [c for c in classes if c == 'citrus_healthy']
    elif pred_class.startswith('mango_'):
        healthy_candidates = [c for c in classes if c == 'mango_healthy']
    else:
        healthy_candidates = [c for c in classes if c.endswith('_healthy')]
    
    if not healthy_candidates:
        # No healthy class found, fall back to confidence-based estimate
        conf = float(probs[pred_idx].item())
        if conf >= 0.85:
            return {'score': 85, 'category': 'Severe', 'ratio': None, 'disease_prob': conf, 'healthy_prob': 0.0, 'confidence': 'High'}
        elif conf >= 0.65:
            return {'score': 65, 'category': 'Moderate', 'ratio': None, 'disease_prob': conf, 'healthy_prob': 0.0, 'confidence': 'Medium'}
        else:
            return {'score': 35, 'category': 'Mild', 'ratio': None, 'disease_prob': conf, 'healthy_prob': 0.0, 'confidence': 'Low'}
    
    healthy_idx = classes.index(healthy_candidates[0])
    disease_prob = float(probs[pred_idx].item())
    healthy_prob = float(probs[healthy_idx].item())
    
    # Compute log-ratio (more numerically stable than simple ratio)
    ratio = disease_prob / (healthy_prob + 1e-8)
    
    # Sigmoid-like mapping: [0, inf] -> [0, 100]
    severity_score = 100 * (1 - 1 / (1 + ratio))
    
    # Categorize by thresholds
    if ratio >= SEVERITY_THRESHOLDS['severe']:
        category = 'Severe'
        conf_level = 'High'
    elif ratio >= SEVERITY_THRESHOLDS['moderate']:
        category = 'Moderate'
        conf_level = 'Medium'
    elif ratio >= SEVERITY_THRESHOLDS['mild']:
        category = 'Mild'
        conf_level = 'Medium'
    else:
        category = 'Early'
        conf_level = 'Low'
    
    return {
        'score': round(severity_score, 1),
        'category': category,
        'ratio': round(ratio, 2),
        'disease_prob': round(disease_prob, 4),
        'healthy_prob': round(healthy_prob, 4),
        'confidence': conf_level,
    }


def _tta_augment(image: Image.Image) -> List[Image.Image]:
    """Generate TTA variants: original, flip, +rotation, -rotation."""
    return [
        image,
        ImageOps.mirror(image),
        image.rotate(8, expand=False),
        image.rotate(-8, expand=False),
    ]


def _predict_with_tta(img_variants: List[Image.Image]) -> Tuple[torch.Tensor, float]:
    """
    Run model on TTA variants and average probabilities.
    Returns averaged probabilities and mean confidence.
    """
    probs_list = []
    
    with torch.no_grad():
        for img in img_variants:
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            logits = MODEL(img_tensor)
            # Apply temperature scaling
            probs = F.softmax(logits / TEMPERATURE, dim=1)
            probs_list.append(probs)
    
    probs_avg = torch.mean(torch.cat(probs_list, dim=0), dim=0)
    return probs_avg, float(probs_avg.max().item())


def _check_family_consistency(probs: torch.Tensor, classes: list) -> Tuple[bool, float, str]:
    """
    Check if citrus% and mango% are well-separated.
    If gap is small, decision is uncertain due to plant ambiguity.
    """
    citrus_prob = sum(probs[i].item() for i, c in enumerate(classes) if c.startswith('citrus_'))
    mango_prob = sum(probs[i].item() for i, c in enumerate(classes) if c.startswith('mango_'))
    
    gap = abs(citrus_prob - mango_prob)
    uncertain = gap < FAMILY_CONSISTENCY_THRESHOLD
    
    family = 'Citrus' if citrus_prob > mango_prob else 'Mango'
    return uncertain, gap, family


print('Loading model...')
MODEL, CLASSES = _load_model_and_classes()
_load_calibration()  # Load temperature & calibration metadata
print(f'Ready — {len(CLASSES)} classes | Temperature: {TEMPERATURE:.4f}')


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
    """
    Single-image prediction with severity level.
    
    Returns:
    - predicted_class: disease or healthy classification
    - confidence: model confidence (0-1)
    - severity: {score, category, ratio, confidence_level}
    - treatment: recommended treatment
    - description: disease description
    """
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = _segment(img)
    
    # TTA inference (average over 4 augmented variants)
    img_variants = _tta_augment(img)
    probs, conf = _predict_with_tta(img_variants)
    
    # Get top prediction
    conf_tta, idx = float(probs.max()), int(probs.argmax())
    pred_class = CLASSES[idx]
    
    # Compute severity
    severity = _compute_severity_score(probs, CLASSES, idx)
    
    # Check family consistency
    uncertain, gap, family = _check_family_consistency(probs, CLASSES)
    
    # Prepare top-5 predictions for debugging
    top5_probs, top5_idx = torch.topk(probs, k=min(5, len(CLASSES)))
    top5_classes = [{'class': CLASSES[int(i.item())], 'prob': float(p.item())} 
                    for p, i in zip(top5_probs, top5_idx)]
    
    return {
        'status': 'success',
        'predicted_class': _display_name(pred_class),
        'raw_class': pred_class,
        'confidence': round(conf_tta, 4),
        'severity': severity,
        'family': family,
        'family_gap': round(gap, 4),
        'uncertain_family': bool(uncertain),
        'treatment': TREATMENTS.get(pred_class, 'No treatment available.'),
        'disease_description': DESCRIPTIONS.get(pred_class),
        'top5_predictions': top5_classes,
        'inference_method': 'TTA (Test-Time Augmentation)',
    }


@app.get('/classes')
def get_classes():
    return {'classes': CLASSES, 'total': len(CLASSES)}


@app.post('/progress')
async def progress(file_before: UploadFile = File(...), file_after: UploadFile = File(...)):
    """
    Detect treatment progress between two photos.
    
    Statuses:
    - 'Healed': Disease changed to corresponding healthy class (same plant family)
    - 'Improved': Severity decreased by > threshold points
    - 'Stable': Change within noise floor (±15 points)
    - 'Worsening': Severity increased by > threshold points
    - 'Unable to measure': Plant families don't match (citrus ↔ mango)
    
    Uses TTA + temperature scaling for robustness.
    """
    # Process before image
    contents_before = await file_before.read()
    img_before = Image.open(io.BytesIO(contents_before)).convert('RGB')
    img_before = _segment(img_before)
    variants_before = _tta_augment(img_before)
    probs_before, conf_before = _predict_with_tta(variants_before)
    idx_before = int(probs_before.argmax())
    class_before = CLASSES[idx_before]
    severity_before = _compute_severity_score(probs_before, CLASSES, idx_before)
    
    # Process after image
    contents_after = await file_after.read()
    img_after = Image.open(io.BytesIO(contents_after)).convert('RGB')
    img_after = _segment(img_after)
    variants_after = _tta_augment(img_after)
    probs_after, conf_after = _predict_with_tta(variants_after)
    idx_after = int(probs_after.argmax())
    class_after = CLASSES[idx_after]
    severity_after = _compute_severity_score(probs_after, CLASSES, idx_after)
    
    # Extract plant family prefix (citrus_, mango_)
    family_before = class_before.split('_')[0]
    family_after = class_after.split('_')[0]
    
    # Check if plant families match
    if family_before != family_after:
        return {
            'status': 'success',
            'progress_status': 'Unable to measure',
            'reason': f'Plant family mismatch: {family_before} → {family_after}',
            'before': {
                'disease': class_before,
                'display_name': _display_name(class_before),
                'severity': severity_before,
                'confidence': round(conf_before, 4),
            },
            'after': {
                'disease': class_after,
                'display_name': _display_name(class_after),
                'severity': severity_after,
                'confidence': round(conf_after, 4),
            },
            'inference_method': 'TTA with threshold-gating',
        }
    
    # Check if healed (disease → healthy of same family)
    if class_after.endswith('_healthy') and not class_before.endswith('_healthy'):
        delta = severity_before['score'] - severity_after['score']
        return {
            'status': 'success',
            'progress_status': 'Healed',
            'delta': round(delta, 1),
            'confidence': 'High',
            'before': {
                'disease': class_before,
                'display_name': _display_name(class_before),
                'severity': severity_before,
                'confidence': round(conf_before, 4),
            },
            'after': {
                'disease': class_after,
                'display_name': _display_name(class_after),
                'severity': severity_after,
                'confidence': round(conf_after, 4),
            },
            'notes': [
                'Leaf returned to healthy state',
                'Disease completely resolved',
            ],
            'inference_method': 'TTA with threshold-gating',
        }
    
    # Standard progress detection (Improved / Stable / Worsening)
    delta = severity_before['score'] - severity_after['score']  # positive = improvement
    improvement_threshold = PROGRESS_THRESHOLDS['improvement_gate']
    
    if delta > improvement_threshold:
        status = 'Improved'
        confidence = 'High' if delta > 25 else 'Medium'
    elif delta < -improvement_threshold:
        status = 'Worsening'
        confidence = 'High' if delta < -25 else 'Medium'
    else:
        status = 'Stable'
        confidence = 'Low'  # Small deltas are below noise floor
    
    return {
        #'status': 'success',
        'progress_status': status,
        
        'delta': round(delta, 1),
        'confidence': confidence,
        'improvement_threshold': improvement_threshold,
        'before': {
            'disease': class_before,
            'display_name': _display_name(class_before),
            'severity': severity_before,
            'confidence': round(conf_before, 4),
        },
        'after': {
            'disease': class_after,
            'display_name': _display_name(class_after),
            'severity': severity_after,
            'confidence': round(conf_after, 4),
        },
        'notes': [
            'Binary signal: only reports change if delta > threshold',
            'Detects large improvements reliably',
            'Early treatment response may be below noise floor (~15 points)',
        ],
        'inference_method': 'TTA with threshold-gating',
        
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
