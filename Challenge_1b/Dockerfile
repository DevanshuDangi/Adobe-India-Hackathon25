# ──────────────────────────────────────────────────────────
# Challenge 1B: Persona‑Driven Document Intelligence Dockerfile
# ──────────────────────────────────────────────────────────

# 1. Base image: specific patch-level tag on Debian Bullseye slim
FROM python:3.10.11-slim-bullseye

# 2. Environment: no .pyc files, unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Set working directory
WORKDIR /app

# 4. Copy application code & input folders into container
COPY . /app

# 5. Install system build dependencies for scikit-learn
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# 6. Upgrade pip and install Python libs (no cache)
RUN pip install --upgrade pip \
 && pip install --no-cache-dir \
      PyMuPDF \
      scikit-learn \
      numpy

# 7. Default command: run your analyzer
CMD ["python", "main.py"]
