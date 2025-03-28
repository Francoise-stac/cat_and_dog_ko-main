# run_mlflow.ps1
# 👉 Script de lancement propre de MLflow Tracking Server

# Aller dans le dossier du projet
Set-Location -Path "C:\Users\Francy\Documents\cat_and_dog_ko-main"

# Créer le dossier artifacts si besoin
if (!(Test-Path -Path ".\artifacts")) {
    New-Item -ItemType Directory -Path ".\artifacts"
}

# Lancer MLflow avec les bons paramètres
# `````````````````````````````````````````````````````````````````````````````````

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5001