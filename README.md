<img src="logo.png">

# 🌿 Plant Disease Detection Using Deep Learning

A multi-class plant disease classification system built with PyTorch and transfer learning, designed for **real-world imbalance**, **interpretability**, and **deployment**.

This project detects diseases from leaf images across citrus and mango crops and now supports:

* Predicted disease
* Confidence score
* Top-k alternative predictions
* Background-removal-assisted inference
* API and Streamlit-based deployment flows

The pipeline is structured to move cleanly from research to production.

---

## Overview

This system uses deep convolutional neural networks with pretrained backbones to classify plant diseases from RGB leaf images.
Rather than simplifying the problem to plant-level classification, the model performs **disease-level diagnosis** for practical usefulness.

The current implementation focuses on:

* Robust performance under severe class imbalance
* Transfer learning for efficiency and accuracy
* Confidence-aware predictions for real-world deployment
* Exportability to backend and UI environments
* Iteration through segmentation and hierarchical-classification experiments

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

### Current Dataset State

The current cleaned split in `data/train` and `data/val` contains **14 active classes** and **21,883 images** total:

* Train: 17,730 images
* Validation: 4,153 images



### Dataset Challenges

* Severe imbalance (some classes have only a few hundred images while others have several thousand)
* Visual similarity between certain diseases
* Mixed image sources and capture conditions
* Legacy splits and duplicated/augmented variants required cleanup

Instead of collapsing labels into plant-level categories, disease-level labels were preserved to maintain diagnostic value.

---

## Model Architecture

Two main model families were explored:

### Prototype

* **ResNet-18** pretrained on ImageNet
* Used to experiment with augmentation and imbalance strategies

### Current Primary Model Family

* **MobileNetV2** pretrained on ImageNet
* Lightweight and better suited for deployment
* Fine-tuned classifier head and later feature blocks
* Exported to TorchScript and ONNX

Additional model variants now in the repo include:

* Fine-tuned MobileNet checkpoints
* Segmentation-focused MobileNet checkpoints
* Crop-specific checkpoints for hierarchical experiments

---

## Training Strategy

### Transfer Learning

* Backbone frozen initially
* Classifier head trained first
* Later feature blocks fine-tuned with lower learning rate

### Configuration

| Component     | Value                                      |
| ------------- | ------------------------------------------ |
| Optimizer     | **Adam** / SGD variants across experiments |
| Loss          | CrossEntropyLoss (class-aware weighting)   |
| Batch size    | 64                                         |
| Epochs        | 10-15                                      |
| LR (head)     | 1e-3                                       |
| LR (backbone) | 1e-4                                       |
| Input size    | 224x224                                    |
| Normalization | ImageNet-stats                             |

---

## Handling Imbalance

The dataset contains extreme class imbalance.
This is handled using a combination of:

* WeightedRandomSampler
* Class-weighted loss
* Rare-class augmentation
* Controlled augmentation strength
* Configurable oversampling controls in `src/config.py`

Rare classes are protected from collapse while avoiding unrealistic synthetic data.

---

## Results

The latest exported metadata in `models/model_metadata.json` reports:

* **Validation accuracy:** `0.9725`
* **Classes:** 14
* **Model version:** 2.0
* **Training date:** 2026-03-16

Strong performance is maintained across both majority and minority classes, though some rare classes remain more difficult than the dominant classes.

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

* > 0.75 -> high confidence
* 0.5-0.75 -> moderate
* < 0.5 -> uncertain (prompt user to retake image)

Low confidence typically indicates:

* Poor lighting
* Background clutter
* Image outside training distribution



---

## Project Structure

```text
plant-disease-detection-using-cnns/
│
├── data/
│   ├── train/                              # Training split
│   ├── val/                                # Validation split
│
├── src/                                    # Source code modules
│   ├── config.py                           # Configuration settings
│   ├── datasets.py                         # Custom dataset classes
│   ├── early_stopping.py                   # Early stopping logic
│   ├── evaluate.py                         # Evaluation functions
│   ├── main.py                             # Training entry point
│   ├── models.py                           # Model architectures
│   ├── train.py                            # Training functions
│   ├── transforms.py                       # Data augmentation
│   └── utils.py                            # Helper functions
│
├── notebooks/                              # Jupyter notebooks
│   ├── 01. data_exploration_and_prototype.ipynb # Prototype
│   │
│   ├── 02. mobilenet_model.ipynb   # Final model's development
│   ├── 03. inference.ipynb         # Infering on real-world examples
│   │
│   ├── 04. image_segmentation.ipynb # Dataset segmentation & cleaning notebook
│   │
│   └── 05. hierarchical_classifier [EXPERIMENT].ipynb 
│
├── scripts/
│   └── rebuild_clean_split.py              # Rebuilds train/val split and checks leakage
│
├── deployment/                             # Deployment package
│   ├── models/
│   │   ├── mobilenet_v2_plant_disease_segmented.pt
│   │   ├── mobilenet_v2_plant_disease_segmented.onnx
│   │   └── model_metadata.json
│   ├── Dev_Guide.md
│   ├── api.py                              # FastAPI web service
│   ├── streamlit_app.py                    # Streamlit demo UI
│   ├── start_ngrok.ps1
│   └── requirements.txt
│
├── models/                                 # Training checkpoints and exports
├── assets/                                 # Demo assets and treatments.json
├── outputs/                                # Experiment outputs
├── inference.py                            # Standalone inference script
├── requirements.txt                        # Project dependencies
└── README.md                               # Project overview
```

---

## Reproducing the Project

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd plant-disease-detection-using-cnns
```

### 2. Create and activate a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install the main project requirements:

```bash
pip install -r requirements.txt
```

If you want to run the API separately, make sure these are available too:

```bash
pip install fastapi uvicorn python-multipart
```

### 4. Prepare the dataset

The training code expects an `ImageFolder`-style layout inside `data/`:

```text
data/
├── train/
│   ├── class_a/
│   ├── class_b/
│   └── ...
└── val/
    ├── class_a/
    ├── class_b/
    └── ...
```

If your split needs rebuilding or leakage cleanup, use:

```bash
python scripts/rebuild_clean_split.py --data-dir data --seed 42
```

This script can:

* detect duplicate files
* regroup augmented variants
* rebuild train/validation splits
* back up the previous split before rewriting it

### 5. Check configuration

Main experiment settings live in `src/config.py`, including:

* dataset paths
* batch size
* rare-class threshold
* oversampling settings
* learning rates
* save paths for checkpoints

Update those values if your local paths or experiment settings differ.

### 6. Train the model

The main training entry point is:

```bash
python -m src.main
```

Training utilities in `src/` handle:

* loading `data/train` and `data/val`
* rare-class-aware augmentation
* weighted sampling
* transfer learning
* early stopping
* checkpoint saving

### 7. Run inference

For a quick local test, edit the image path inside `inference.py` and run:

```bash
python inference.py
```

### 8. Run the deployed interfaces

**FastAPI backend**

```bash
uvicorn deployment.api:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

* `http://localhost:8000/`
* `http://localhost:8000/docs`

**Streamlit app**

```bash
streamlit run deployment/streamlit_app.py
```

### 9. Use the exported models

The repository includes deployable artifacts in `models/` and `deployment/models/`, including:

* `.pth` training checkpoints
* `.pt` TorchScript exports
* `.onnx` exports
* metadata and calibration JSON files

---

## Inference

Example usage:

```python
classifier = PlantDiseaseClassifier(r"models/mobilenet_v2_plant_disease.pt")
result = classifier.predict(r"path_to_image.jpg")
print(result)
```

The current inference flow may include:

* Background removal using `rembg`
* Resize to `224x224`
* Tensor conversion and normalization
* Softmax probabilities for confidence-aware output

---

## Deployment

The project can now be run in multiple ways:

* Standalone inference with `inference.py`
* **FastAPI** backend via `deployment/api.py`
* **Streamlit** UI via `deployment/streamlit_app.py`

The API can also return treatment suggestions sourced from `assets/treatments.json`.

---

## Export

The model can be exported to:

* TorchScript (`.pt`)
* ONNX (`.onnx`)

Metadata now also includes:

* class names
* normalization
* input size
* validation accuracy
* model version
* training date

---

## Limitations

* Some classes still have limited samples
* Performance may drop on real field images
* Labels depend on dataset annotations
* The repo still contains some legacy 15-class artifacts alongside the current 14-class exports
* Different experiments use slightly different preprocessing/model-loading paths

These are acknowledged as part of the research and deployment scope.

---

## Future Work

* Collect more expert-verified images
* Improve rare-class coverage
* Continue confidence calibration
* Refine segmentation-assisted inference
* Expand hierarchical classification experiments
* Expand to additional crops

---

## Conclusion

This project shows that transfer learning with careful fine-tuning can achieve strong performance on complex, imbalanced plant disease datasets.

The resulting system is suitable as:

* A research reference
* A teaching example
* A foundation for deployable agricultural tools

---
