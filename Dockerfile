FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

WORKDIR /IndexMind

# ---------------------------------------------------------
# 🧰 System packages: Python + build tools + all key libs
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-distutils python3-pip \
    build-essential git curl wget unzip \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libjpeg-dev libpng-dev libtiff-dev \
    libopenblas-dev liblapack-dev \
    libffi-dev libssl-dev \
    poppler-utils tesseract-ocr ghostscript \
    ffmpeg imagemagick \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
	
	
RUN pip install --no-cache-dir --break-system-packages uv

COPY . .

RUN uv pip install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cu126 --system
RUN uv pip install --break-system-packages -r requirements.txt --system

EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]

