"""
Streamlit UI for Plant Disease Classification
Run: streamlit run streamlit_app.py
"""

from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from rembg import remove
from torchvision import transforms


CLASSES = [
    "citrus_black_spot",
    "citrus_canker",
    "citrus_foliage_damage",
    "citrus_greening",
    "citrus_healthy",
    "citrus_mealybugs",
    "citrus_melanose",
    "mango_anthracnose",
    "mango_bacterial_canker",
    "mango_cutting_weevil",
    "mango_die_back",
    "mango_gall_midge",
    "mango_healthy",
    "mango_powdery_mildew",
    "mango_sooty_mould",
]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.502275, 0.528588, 0.474627], std=[0.251866, 0.251658, 0.335502]),
    ]
)


@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "models" / "mobilenet_v2_plant_disease_segmented.pt"
    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()
    return model


def segment_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    try:
        segmented = remove(image)
        if segmented.mode != "RGBA":
            return segmented.convert("RGB")

        canvas = Image.new("RGB", segmented.size, (0, 0, 0))
        canvas.paste(segmented, mask=segmented.getchannel("A"))
        return canvas
    except Exception:
        return image


def predict(image: Image.Image, model: torch.jit.ScriptModule):
    image = segment_image(image)
    image_tensor = TRANSFORM(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1).squeeze()

    conf, idx = torch.max(probs, dim=0)
    return CLASSES[idx.item()], round(float(conf.item()), 2)


def main():
    st.set_page_config(page_title="Plant Disease Detector", layout="centered")
    st.title("Plant Disease Detector")

    model = load_model()
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", width=200)

        if st.button("Predict", type="primary"):
            with st.spinner("Running inference..."):
                predicted_class, confidence = predict(image, model)
            st.success(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence}")


if __name__ == "__main__":
    main()
