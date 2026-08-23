FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY main.py .
COPY yolov8m.pt .
COPY osnet_x0_25_msmt17.pt .

RUN mkdir -p /app/input /app/output

CMD ["python", "main.py"]