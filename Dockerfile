# Stage 1: Build the React Application
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend

# Copy dependency definitions
COPY web/frontend/package.json web/frontend/package-lock.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY web/frontend/ ./

# Build the app
RUN npm run build


# Stage 2: Build the Python Backend and serve
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if needed (e.g. for numpy/matplotlib optimized build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements/project
COPY pyproject.toml requirements.txt ./
COPY src/ ./src/
COPY web/backend/ ./web/backend/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir .
RUN pip install --no-cache-dir fastapi uvicorn matplotlib "uvicorn[standard]"

# Copy built frontend assets from Stage 1 to a directory FastAPI can serve
# In app.py we check for 'static' relative to app.py location
# app.py is in /app/web/backend/app.py. So 'static' should be /app/web/backend/static
COPY --from=frontend-builder /app/frontend/dist /app/web/backend/static

# Expose port (Render uses environment variable PORT, typically 10000, but we can default)
ENV PORT=8000
EXPOSE $PORT

# Run the application
# We run from /app root so imports like 'web.backend.app' work?
# No, app.py expects to be run either as module or script.
# "python web/backend/app.py" uses uvicorn.run internally.
# Better to use uvicorn command line for production to control workers etc.
# "uvicorn web.backend.app:app --host 0.0.0.0 --port $PORT"
CMD uvicorn web.backend.app:app --host 0.0.0.0 --port $PORT
