# Use the official Python image as base
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project into the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y sqlite3 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Ensure the persistent data directory exists
RUN mkdir -p /app/data

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production
ENV DATABASE_PATH=/app/data/mlflow.db
ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5001

# Expose Flask (5000) and MLflow (5001) ports
EXPOSE 5000 5001

# Initialize the SQLite database if it doesn't exist
RUN sqlite3 /app/data/mlflow.db "PRAGMA journal_mode=WAL;"

# Command to start MLflow and Flask (ensures MLflow starts first)
CMD ["sh", "-c", "mlflow server --backend-store-uri sqlite:////app/data/mlflow.db --default-artifact-root /app/artifacts --host 0.0.0.0 --port 5001 & sleep 5 && python app.py"]