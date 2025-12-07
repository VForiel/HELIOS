FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for matplotlib and other scientific packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY web/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy HELIOS source code and install it
COPY src/ /app/src/
COPY pyproject.toml /app/
COPY README.md /app/
# Install HELIOS (assuming it's installable via pip install .)
RUN pip install .

# Copy backend application code
COPY web/backend/ /app/backend/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
