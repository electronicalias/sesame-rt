#!/bin/bash

echo "WARNING: This will delete ALL Docker data, including containers, images, volumes, networks, and build cache."
read -p "Are you sure you want to proceed? (y/N): " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Operation canceled."
    exit 1
fi

echo "Stopping all running containers..."
docker stop $(docker ps -q)

echo "Removing all containers..."
docker rm -f $(docker ps -aq)

echo "Removing all images..."
docker rmi -f $(docker images -q)

echo "Removing all volumes..."
docker volume rm -f $(docker volume ls -q)

echo "Removing all networks (except default)..."
docker network prune -f

echo "Pruning system..."
docker system prune -af --volumes

echo "Removing Docker build cache..."
docker builder prune -af

echo "Reset complete!"