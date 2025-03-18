# Utiliser une image Python 3.11 comme base
FROM python:3.11

# Définir le dossier de travail
WORKDIR /app

# Copier tous les fichiers du projet dans le conteneur
COPY . /app

# Installer les dépendances
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install mlflow
# Installer les dépendances
# RUN pip install --upgrade pip && \
#     pip install -r requirements.txt mlflow

# Exposer les ports Flask (5000) et MLflow (5001)
EXPOSE 5000 5001

# Lancer MLflow en arrière-plan puis démarrer Flask
CMD mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5001 & python app.py
