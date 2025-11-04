#!/bin/bash

IMAGE_NAME="index-mind"
CONTAINER_NAME="indexmind_container"

echo "🔨 Building Docker image: $IMAGE_NAME ..."
docker build -t $IMAGE_NAME .

# Check if NVIDIA GPU is available
if docker info | grep -q "Runtimes:.*nvidia"; then
    echo "🚀 Starting container with GPU support..."
    docker run --rm -it \
        --gpus all \
        -p 8000:8000 \
        --name $CONTAINER_NAME \
        $IMAGE_NAME
else
    echo "⚠️  No NVIDIA runtime detected. Starting container without GPU..."
    docker run --rm -it \
        -p 8000:8000 \
        --name $CONTAINER_NAME \
        $IMAGE_NAME
fi
