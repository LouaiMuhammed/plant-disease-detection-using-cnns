# Plant Disease Model - Developer Handoff

## Quick Start for Backend

```bash
cd deployment
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Visit: http://localhost:8000/docs

## Quick Start for Flutter

### Using Backend API (Recommended)

1. **Install package:**
```yaml
dependencies:
  http: ^1.1.0
```

2. **Make prediction:**
```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<Map<String, dynamic>> predictDisease(File image) async {
  var request = http.MultipartRequest(
    'POST', 
    Uri.parse('http://YOUR_BACKEND_URL/predict')
  );
  
  request.files.add(
    await http.MultipartFile.fromPath('file', image.path)
  );
  
  var response = await request.send();
  var json = jsonDecode(await response.stream.bytesToString());
  
  return json;
  // {"predicted_class": "citrus_canker", "confidence": 0.985}
}
```

3. **Display result:**
```dart
var result = await predictDisease(imageFile);
print('Disease: ${result['predicted_class']}');
print('Confidence: ${(result['confidence'] * 100).toStringAsFixed(1)}%');
```

## Disease Classes (15 total)

**Citrus (7):**
- citrus_black_spot
- citrus_canker
- citrus_foliage_damage
- citrus_greening
- citrus_healthy
- citrus_mealybugs
- citrus_melanose

**Mango (8):**
- mango_anthracnose
- mango_bacterial_canker
- mango_cutting_weevil
- mango_die_back
- mango_gall_midge
- mango_healthy
- mango_powdery_mildew
- mango_sooty_mould

## Image Requirements

- Format: JPG, PNG
- Any size (will be resized to 224×224)
- Clear photo of leaf showing symptoms
- Good lighting

## Model Performance

- Accuracy: 95.15%
- Average Confidence: 92.46%
- Inference Speed: ~500ms on CPU
- Model Size: 9 MB

## API Endpoints

### POST /predict
Upload image and get prediction

**Request:**
```bash
curl -X POST -F "file=@leaf.jpg" http://localhost:8000/predict
```

**Response:**
```json
{
  "predicted_class": "citrus_canker",
  "confidence": 0.985
}
```

### GET /classes
Get all disease classes

**Response:**
```json
{
  "classes": ["citrus_black_spot", "citrus_canker", ...]
}
```

### GET /
Health check

**Response:**
```json
{
  "status": "online",
  "model": "loaded"
}
```

## Preprocessing Details

**IMPORTANT:** If running model directly (not via API), preprocessing must match training:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])
```

## Troubleshooting

**Backend:**
- "Model file not found" → Copy `mobilenet_v2_plant_disease.pt` to `models/` folder
- "Import errors" → Run `pip install -r requirements.txt`
- Port 8000 in use → Change port: `uvicorn api:app --port 8001`

**Flutter:**
- Connection refused → Check backend is running
- CORS errors → Backend allows all origins by default
- Slow response → Expected on first request (model loading)

## Support

For questions, contact: louaimuhammed@gmail.com

Model version: 1.0
Last updated: February 2026