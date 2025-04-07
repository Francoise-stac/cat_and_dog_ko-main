# # Use the official Python image as base
# FROM python:3.11-slim

# # Set the working directory inside the container
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends sqlite3 && \
#     rm -rf /var/lib/apt/lists/*

# # Copy requirements first for better caching
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application
# COPY . .

# # Create directories for persistent data
# # RUN mkdir -p /app/data /app/artifacts && \
# #     chown -R 1000:1000 /app/data /app/artifacts
# # RUN mkdir -p /app/data /app/artifacts \
# #     && mkdir -p /app/logs \
# #     && chown -R 1000:1000 /app/data /app/artifacts /app/logs
# RUN mkdir -p /app/data /app/artifacts /app/logs /app/instance && \
#     chown -R 1000:1000 /app/artifacts /app/logs /app/instance /app/models



# # Set environment variables
# ENV FLASK_APP=app.py \
#     FLASK_RUN_HOST=0.0.0.0 \
#     FLASK_ENV=production \
#     DATABASE_PATH=/app/data/mlflow.db \
#     MLFLOW_TRACKING_URI=http://127.0.0.1:5001

# # Expose ports
# EXPOSE 5000 5001

# # Set non-root user
# USER 1000

# # Command to start MLflow and Flask
# CMD ["sh", "-c", "mlflow server --backend-store-uri sqlite:////app/data/mlflow.db --default-artifact-root /app/artifacts --host 0.0.0.0 --port 5001 & sleep 5 && python app.py"]



# Use the official Python image as base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends sqlite3 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create required folders and set permissions
RUN mkdir -p /app/artifacts /app/logs /app/instance /app/models && \
    chown -R 1000:1000 /app/artifacts /app/logs /app/instance /app/models
RUN mkdir -p /app/mlruns_artifacts && \
    chown -R 1000:1000 /app/mlruns_artifacts

# Set environment variables
ENV FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_ENV=production \
    MLFLOW_TRACKING_URI=http://127.0.0.1:5001

# Expose Flask and MLflow ports
EXPOSE 5000 5001

# Use non-root user to avoid permission issues
USER 1000

# Start MLflow server and then launch Flask

CMD ["sh", "-c", "mlflow server --backend-store-uri sqlite:////app/data/mlflow.db --default-artifact-root /app/artifacts --host 0.0.0.0 --port 5001 & sleep 5 && python app.py"]



