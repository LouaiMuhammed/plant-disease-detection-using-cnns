FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.deploy.txt ./requirements.deploy.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.deploy.txt

COPY deployment/api.py ./api.py
COPY src ./src
COPY assets/treatments.json ./assets/treatments.json
COPY assets/diseases_description.json ./assets/diseases_description.json
COPY deployment/models/mobilenet_v2_plant_disease_segmented.pt ./deployment/models/mobilenet_v2_plant_disease_segmented.pt
COPY deployment/models/model_metadata.json ./deployment/models/model_metadata.json

EXPOSE 8000

