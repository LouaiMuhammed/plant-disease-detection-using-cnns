FROM pytorch/pytorch:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install \
        rembg==2.0.67 \
        fastapi==0.129.0 \
        uvicorn==0.40.0 \
        pytorch-cpu \
        pillow==12.0.0

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
