<img src="logo.png">

# 🌿 Plant Disease Detection Using Deep Learning

A multi-class plant disease classification system built with PyTorch and transfer learning, designed for **real-world imbalance**, **interpretability**, and **mobile deployment**.

This project detects diseases from leaf images across citrus and mango crops and returns:

* Predicted disease
* Confidence score
* Top-k alternative predictions

The pipeline is structured to move cleanly from research to production.

---

## Overview

This system uses deep convolutional neural networks with pretrained backbones to classify plant diseases from RGB leaf images.
Rather than simplifying the problem to plant-level classification, the model performs **disease-level diagnosis** for practical usefulness.

The final implementation focuses on:

* Robust performance under severe class imbalance
* Transfer learning for efficiency and accuracy
* Confidence-aware predictions for real-world deployment
* Exportability to mobile and backend environments

---

## Dataset

Images are organized using the standard `ImageFolder` directory format.

### Crops and Diseases

**Citrus**

* Healthy
* Canker
* Greening
* Black spot
* Melanose
* Mealybugs
* Foliage damage

**Mango**

* Healthy
* Anthracnose
* Powdery mildew
* Die back
* Bacterial canker
* Gall midge
* Cutting weevil
* Sooty mould

### Dataset Challenges

* Severe imbalance (some classes <50 images, others >10k)
* Visual similarity between certain diseases
* Mixed image sources and capture conditions

Instead of collapsing labels into plant-level categories, disease-level labels were preserved to maintain diagnostic value.

---

## Model Architecture

Two models were explored:

### Prototype

* **ResNet-18** pretrained on ImageNet
* Used to experiment with augmentation and imbalance strategies

### Final Model

* **MobileNetV2** pretrained on ImageNet
* Lightweight and optimized for mobile deployment
* Fine-tuned classifier head and final convolutional block

---

## Training Strategy

### Transfer Learning

* Backbone frozen initially
* Classifier head trained first
* Final convolutional block fine-tuned with lower learning rate

### Configuration

| Component     | Value                             |
| ------------- | --------------------------------- |
| Optimizer     | Adam                              |
| Loss          | CrossEntropyLoss (class-weighted) |
| Batch size    | 64                                |
| Epochs        | 10–15                             |
| LR (head)     | 1e-3                              |
| LR (backbone) | 1e-4                              |
| Input size    | 224×224                           |
| Normalization | ImageNet mean/std                 |

---

## Handling Imbalance

The dataset contains extreme class imbalance.
This was addressed using a combination of:

* WeightedRandomSampler
* Class-weighted loss
* Rare-class augmentation
* Controlled augmentation strength

Rare classes were protected from collapse while avoiding unrealistic synthetic data.

---

## Results

**Validation accuracy:** ~95–96%
**Macro F1:** ~0.94–0.96

Strong performance across both majority and minority classes.

Rare disease classes retained usable recall rather than collapsing into majority predictions.

---

## Model Interpretability

Grad-CAM was used to visualize attention regions.

Findings:

* Model focuses on lesion areas and discoloration
* Errors occur mainly between visually similar diseases
* Predictions are generally explainable and biologically plausible

This improves trustworthiness and debugging ability.

---

## Confidence-Aware Predictions

The model outputs a probability distribution over classes.

Applications should interpret confidence as:

* > 0.75 → high confidence
* 0.5–0.75 → moderate
* <0.5 → uncertain (prompt user to retake image)

Low confidence typically indicates:

* Poor lighting
* Background clutter
* Image outside training distribution

---

## Project Structure

```
plant-disease-detection-using-cnns/
│
├── data/                                   
│   ├── train/                              # Training images (24,499 images)
│   │   ├── citrus_black_spot/             # 136 images
│   │   ├── citrus_canker/                 # 8,998 images
│   │   ├── citrus_foliage_damage/         # 1,680 images
│   │   ├── citrus_greening/               # 163 images
│   │   ├── citrus_healthy/                # 5,107 images
│   │   ├── citrus_mealybugs/              # 3,135 images
│   │   ├── citrus_melanose/               # 2,080 images
│   │   ├── mango_anthracnose/             # 400 images
│   │   ├── mango_bacterial_canker/        # 400 images
│   │   ├── mango_cutting_weevil/          # 400 images
│   │   ├── mango_die_back/                # 400 images
│   │   ├── mango_gall_midge/              # 400 images
│   │   ├── mango_healthy/                 # 400 images
│   │   ├── mango_powdery_mildew/          # 400 images
│   │   └── mango_sooty_mould/             # 400 images
│   │
│   └── val/                                # Validation images (6,127 images)
│       ├── citrus_black_spot/
│       ├── citrus_canker/
│       ├── mango_sooty_mould/
│       └── (same structure as train/)
│
├── src/                                     # Source code modules
│   ├── __init__.py                         # Package init
│   ├── config.py                           # Configuration settings
│   ├── datasets.py                         # Custom dataset classes
│   ├── early_stopping.py                   # Early stopping logic
│   ├── evaluate.py                         # Evaluation functions
│   ├── models.py                           # Model architectures
│   ├── train.py                            # Training functions
│   ├── transforms.py                       # Data augmentation
│   └── utils.py                            # Helper functions
│
├── notebooks/                               # Jupyter notebooks
│   ├── 01. data_exploration_and_prototype.ipynb          # Prototype            
│   └── 02. mobilenet_model.ipynb        # Deployed Model
│   └── 03. hierarchical_classifier.ipynb        # Separation Experiement
│
│
├── deployment/                              # Deployment package for developers
│   ├── models/
│   │   ├── mobilenet_v2_plant_disease.pt   # Copy of TorchScript model
│   │   ├── mobilenet_v2_plant_disease.onnx # Copy of ONNX model
│   │   └── model_metadata.json             # Copy of metadata
│   │
│   ├── Dev_Guide.md
│   ├── api.py                              # FastAPI web service
│   └── examples/                           # (optional) Test images
│
├── venv/                                    # Python virtual environment (optional)
│   └── (Python packages installed here)
│
├── requirements.txt                         # Project dependencies
└── README.md                               # Project overview
```

---

## Inference

Example usage:

```python
img = Image.open(r"path").convert("RGB")
img_tensor = get_val_transform()(img)

with torch.no_grad():
    logits = model(img_tensor.unsqueeze(0).to(device))
    probs = F.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)

print(f"{classes[pred.item()]}: {conf.item():.2%}")
```

---

## Export

The model can be exported to:

* TorchScript (`.pt`) for mobile deployment
* ONNX (`.onnx`) for cross-platform inference

Metadata includes:

* class names
* normalization
* input size

---

## Limitations

* Some classes still have limited samples
* Performance may drop on real field images
* Labels depend on dataset annotations
* Rare disease diversity is limited

These are acknowledged as part of the research scope.

---

## Future Work

* Collect expert-verified images
* Improve rare-class coverage
* Add confidence calibration
* Quantize model for faster mobile inference
* Expand to additional crops

---

## Conclusion

This project shows that transfer learning with careful fine-tuning can achieve strong performance on complex, imbalanced plant disease datasets.

The resulting model is suitable as:

* A research reference
* A teaching example
* A foundation for mobile agricultural tools

---

