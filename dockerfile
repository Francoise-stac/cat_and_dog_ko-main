# Utiliser une image Python 3.11 comme base
FROM python:3.11

# Définir le dossier de travail
WORKDIR /app

# Copier tous les fichiers du projet dans le conteneur
COPY . /app

RUN mkdir -p /tmp/mlflow_tmp
ENV MLFLOW_TEMP_DIR=/tmp/mlflow_tmp

# Installer les dépendances
RUN pip install --upgrade pip && \
    pip install -r requirements.txt mlflow

RUN mkdir -p data/retraining/chat data/retraining/chien

# Exposer les ports Flask (5000) et MLflow (5001)
EXPOSE 5000 5001

# Lancer MLflow en arrière-plan puis démarrer Flask
# CMD mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5001 & python app.py
# CMD ["sh", "-c", "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5001 & sleep 5 && python app.py"]

CMD ["sh", "-c", "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5001 & sleep 5 && exec python app.py"]

