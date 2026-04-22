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
        pillow==12.0.0 && \
    pip install \
        torch==2.9.1 \
        --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
