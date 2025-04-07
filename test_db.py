import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy


# 🔧 Utilise un chemin absolu pour l'instance_path
instance_path = os.path.abspath("instance")

app = Flask(__name__, instance_path=instance_path)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///mlflow.db"
db = SQLAlchemy(app)

with app.app_context():
    try:
        db.create_all()
        print("✅ Connexion à la base de données réussie et tables créées")
    except Exception as e:
        print("❌ Erreur :", e)