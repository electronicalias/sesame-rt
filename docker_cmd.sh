#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t streaming-asr .

# Run the container with GPU access
echo "Starting container with GPU access..."
docker run --gpus all -p 8000:8000 -it streaming-asr

# If you don't have GPU, use this command instead:
# docker run -p 8000:8000 -it streaming-asr