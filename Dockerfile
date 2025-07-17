# ✅ Use Python 3.10 to allow langchain-core + pydantic v1
FROM python:3.10-slim

# System dependencies for audio
RUN apt-get update && apt-get install -y \
    portaudio19-dev ffmpeg build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools==67.7.2
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Default command (adjust as needed)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]