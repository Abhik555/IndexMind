#!/bin/bash

IMAGE_NAME="index-mind"
CONTAINER_NAME="indexmind_container"

echo "🚀 Starting container from image: $IMAGE_NAME"

# Check for NVIDIA runtime availability
if docker info | grep -q "Runtimes:.*nvidia"; then
    echo "✅ NVIDIA runtime detected — running with GPU support..."
    docker run --rm -it \
        --gpus all \
        -p 8000:8000 \
        --name "$CONTAINER_NAME" \
        "$IMAGE_NAME"
else
    echo "⚠️  NVIDIA runtime not found — running without GPU support..."
    docker run --rm -it \
        -p 8000:8000 \
        --name "$CONTAINER_NAME" \
        "$IMAGE_NAME"
fi
