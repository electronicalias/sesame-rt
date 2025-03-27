FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy ONLY requirements files first
COPY requirements.txt .

# Install dependencies (this layer will be cached unless requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install websockets
RUN pip install transformers

# THEN copy the rest of your application code
COPY . .

# Copy application files
COPY app.py .
RUN mkdir -p static

# Add SSL
COPY ssl/ /app/ssl/

# Create default index.html if needed (will be auto-created by the app)

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]