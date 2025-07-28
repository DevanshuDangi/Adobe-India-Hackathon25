# Use Python 3.10 on amd64 architecture
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyMuPDF (fitz) and scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file if using pip freeze (optional)
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Install Python dependencies directly
RUN pip install --no-cache-dir \
    PyMuPDF==1.22.5 \
    numpy==1.26.4 \
    scikit-learn==1.4.2

# Copy the processing script into container
COPY process_pdfs.py .

# Default command to run the processing script
CMD ["python", "process_pdfs.py"]
