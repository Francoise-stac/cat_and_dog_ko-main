from flask import Flask, request, render_template, redirect, url_for, flash, session
from functools import wraps
import numpy as np
from keras.models import load_model
import os
from io import BytesIO
from PIL import Image
import base64
import mlflow
from datetime import datetime
from keras import backend as K
import keras
from datab import db, Feedback, User
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from keras.preprocessing import image
from glob import glob
import tensorflow as tf
from dotenv import load_dotenv
from flask import send_from_directory
import matplotlib.pyplot as plt
import os


os.environ["MLFLOW_TEMP_DIR"] = "/tmp/mlflow_tmp"

load_dotenv()

# Paramètres globaux
SEED = 42
IMAGE_SIZE = 128
MAX_IMG = 4000

# Configuration MLflow conditionnelle
TESTING = os.environ.get('TESTING', 'False') == 'True'

if not TESTING:
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5001'))
    mlflow.set_experiment("feedback_experiment")

app = Flask(__name__)
app.secret_key = "une_clé_secrète_pour_session_et_flash"
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'votre_clé_secrète_par_défaut')

# Modifier la configuration de la base de données
if TESTING:
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///mlflow.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
print("Base de données configurée :", app.config["SQLALCHEMY_DATABASE_URI"])

db.init_app(app)
migrate = Migrate(app, db)

with app.app_context():
    db.create_all()

# # Charger le modèle


# MODEL_PATH = r"C:\Users\Francy\Documents\cat_and_dog_ko-main\models\model.keras"
# # MODEL_PATH = os.path.join(os.getcwd(), "models", "model.keras")
# Définir le chemin correct du modèle pour Docker
MODEL_PATH = "models/model.keras"

# Vérifier si le fichier existe avant de le charger
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le fichier modèle n'existe pas : {MODEL_PATH}")

print(f"Chargement du modèle depuis : {MODEL_PATH}")
model = load_model(MODEL_PATH)
model.make_predict_function()


def model_predict(img, model):
    if model is None:
        return [[0.7]]  # Valeur de test par défaut
    img = img.resize((128, 128))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    return preds


def should_retrain_model(threshold=5):
    retraining_data_path = "data/retraining"
    num_images = sum([len(files) for r, d, files in os.walk(retraining_data_path)])
    return num_images >= threshold




def plot_history(history, output_path="training_plot.png"):
    plt.figure(figsize=(10, 4))

    # Sous-plot 1 : Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Courbe de loss')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.legend()

    # Sous-plot 2 : Accuracy
    plt.subplot(1, 2, 2)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Courbe de précision')
    plt.xlabel('Époque')
    plt.ylabel('Accuracy')
    plt.legend()

    # Sauvegarder le graphique
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# def retrain_model():
#     print("Réentraînement du modèle...")
#     IMAGE_SIZE = 128
#     data_dir = "data/retraining"

#     # Initialisation des listes pour les images et les labels
#     X = []
#     y = []

#     for index_animal, animal in enumerate(["chat", "chien"]):
#         for img_path in glob(os.path.join(data_dir, animal, "*.jpg")):
#             try:
#                 # Chargement de l'image
#                 img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#                 # Conversion en tableau Numpy
#                 img_array = image.img_to_array(img)
#                 # Ajout de l'image et du label à la liste
#                 X.append(img_array)
#                 y.append(index_animal)
#             except Exception as e:
#                 print(f"Erreur lors du chargement de l'image {img_path}: {e}")

#     # Conversion des listes en tableaux Numpy
#     X = np.array(X)  # Forme (n_samples, 128, 128, 3)
#     y = np.array(y)  # Forme (n_samples,)

#     # Normalisation des images (valeurs des pixels entre 0 et 1)
#     X = X / 255.0

#     # Encodage des labels en one-hot encoding
#     y = keras.utils.to_categorical(y, num_classes=2)

#     # Vérification des formes des données
#     print(f"Forme de X : {X.shape}")
#     print(f"Forme de y : {y.shape}")

#     # Charger l'ancien modèle
#     model = load_model(MODEL_PATH)

#     # Réentraîner le modèle avec les nouvelles données
#     model.fit(X, y, epochs=5, batch_size=32)

#     # Sauvegarder le modèle mis à jour
#     model.save(MODEL_PATH)
#     print(f"Modèle réentraîné et sauvegardé dans : {MODEL_PATH}")

#     # Créer une nouvelle version du modèle dans MLflow
#     with mlflow.start_run():
#         mlflow.keras.log_model(
#             model, "model", registered_model_name="Model_for_User_Feedback"
#         )
#         print("Modèle réentraîné sauvegardé dans MLflow.")
#     mlflow.end_run()


# def retrain_model():
#     print("Réentraînement du modèle...")
#     IMAGE_SIZE = 128
#     data_dir = "data/retraining"

#     X = []
#     y = []

#     for index_animal, animal in enumerate(["chat", "chien"]):
#         for img_path in glob(os.path.join(data_dir, animal, "*.jpg")):
#             try:
#                 img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#                 img_array = image.img_to_array(img)
#                 X.append(img_array)
#                 y.append(index_animal)
#             except Exception as e:
#                 print(f"Erreur lors du chargement de l'image {img_path}: {e}")

#     X = np.array(X)
#     y = np.array(y)

#     X = X / 255.0
#     y = keras.utils.to_categorical(y, num_classes=2)

#     print(f"Forme de X : {X.shape}")
#     print(f"Forme de y : {y.shape}")

#     model = load_model(MODEL_PATH)

#     # Réentraînement
#     history = model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

#     model.save(MODEL_PATH)
#     print(f"Modèle réentraîné et sauvegardé dans : {MODEL_PATH}")

#     # Exemple d’entrée pour la signature MLflow
    
#     input_example = np.random.rand(1, 128, 128, 3).astype(np.float32)

#     # S’assurer que le dossier existe
#     os.makedirs("mlflow_logs", exist_ok=True)
#     np.save("mlflow_logs/input_example.npy", input_example)  # Juste pour test

#     with mlflow.start_run():
#         # Log du modèle avec signature
#         mlflow.keras.log_model(
#             model,
#             "model",
#             registered_model_name="Model_for_User_Feedback",
#             input_example=input_example
#         )

#         # Log des métriques par époque
#         for epoch in range(len(history.history['loss'])):
#             mlflow.log_metric("loss", history.history['loss'][epoch], step=epoch)
#             if 'accuracy' in history.history:
#                 mlflow.log_metric("accuracy", history.history['accuracy'][epoch], step=epoch)
#             if 'val_loss' in history.history:
#                 mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
#             if 'val_accuracy' in history.history:
#                 mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)

#         # Générer et logger une image de la courbe d'apprentissage
#         plot_history(history, "training_plot.png")
#         mlflow.log_artifact("training_plot.png")

#         # Log de l'historique complet
#         print("Modèle réentraîné et métriques enregistrées dans MLflow.")

#     mlflow.end_run()

def retrain_model():
    print("Réentraînement du modèle...")
    IMAGE_SIZE = 128
    data_dir = "data/retraining"

    X = []
    y = []

    for index_animal, animal in enumerate(["chat", "chien"]):
        for img_path in glob(os.path.join(data_dir, animal, "*.jpg")):
            try:
                img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
                img_array = image.img_to_array(img)
                X.append(img_array)
                y.append(index_animal)
            except Exception as e:
                print(f"Erreur lors du chargement de l'image {img_path}: {e}")

    # Conversion des listes en tableaux Numpy
    X = np.array(X) # Forme (n_samples, 128, 128, 3)
    y = np.array(y) # Forme (n_samples,)

    # Normalisation des images (valeurs des pixels entre 0 et 1)
    X = X / 255.0
    y = keras.utils.to_categorical(y, num_classes=2)

    print(f"Forme de X : {X.shape}")
    print(f"Forme de y : {y.shape}")

    model = load_model(MODEL_PATH)

    # Réentraînement
    history = model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

    model.save(MODEL_PATH)
    print(f"✅ Modèle réentraîné et sauvegardé dans : {MODEL_PATH}")

    # Tentative de log du modèle dans MLflow
    input_example = np.random.rand(1, 128, 128, 3).astype(np.float32)

    with mlflow.start_run():
        try:
            from mlflow.models.signature import infer_signature
            signature = infer_signature(input_example, model.predict(input_example))

            mlflow.keras.log_model(
                model,
                "model",
                registered_model_name="Model_for_User_Feedback",
                input_example=input_example,
                signature=signature
            )
            print("✅ Modèle loggé avec input_example et signature.")
        except Exception as e:
            print(f"⚠️ Erreur lors du log avec input_example : {e}")
            print("📦 Log du modèle sans input_example...")
            mlflow.keras.log_model(
                model,
                "model",
                registered_model_name="Model_for_User_Feedback"
            )
            print("✅ Modèle loggé sans input_example.")

        # Log des métriques par époque
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("loss", history.history['loss'][epoch], step=epoch)
            if 'accuracy' in history.history:
                mlflow.log_metric("accuracy", history.history['accuracy'][epoch], step=epoch)
            if 'val_loss' in history.history:
                mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            if 'val_accuracy' in history.history:
                mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)

        # Générer et logger une image de la courbe d'apprentissage
        plot_history(history, "training_plot.png")
        mlflow.log_artifact("training_plot.png")

        print("📊 Courbe et métriques enregistrées dans MLflow.")

    mlflow.end_run()


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Veuillez vous connecter pour accéder à cette page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function



@app.route("/rejected_images")
def rejected_images():
    image_dir = "data/retraining"
    image_data = []

    for label in ["chat", "chien"]:
        folder_path = os.path.join(image_dir, label)
        if os.path.exists(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(".jpg"):
                    image_data.append({
                        "label": label,
                        "filename": file_name,
                        "filepath": f"/rejected_image/{label}/{file_name}"
                    })

    return render_template("rejected_images.html", images=image_data)

@app.route("/rejected_image/<label>/<filename>")
def serve_rejected_image(label, filename):
    directory = os.path.join("data/retraining", label)
    return send_from_directory(directory, filename)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        
        if user is None:
            flash("Nom d'utilisateur incorrect", "danger")
            return redirect(url_for('login'))

        if not user.check_password(password):
            flash("Mot de passe incorrect", "danger")
            return redirect(url_for('login'))
        
        session['user_id'] = user.id
        flash('Connexion réussie!', 'success')
        return redirect(url_for('home'))

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Ce nom d\'utilisateur existe déjà', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Cet email est déjà utilisé', 'error')
            return render_template('register.html')
        
        user = User(username=username, email=email)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Inscription réussie! Vous pouvez maintenant vous connecter.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Vous êtes déconnecté.', 'info')
    return redirect(url_for('login'))

@app.route("/", methods=["GET"])
@login_required
def home():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
@login_required
def upload():
    if "file" not in request.files:
        flash("Aucun fichier envoyé.", "error")
        return redirect(url_for("home"))

    f = request.files["file"]
    buffered_img = BytesIO(f.read())
    img = Image.open(buffered_img)
    base64_img = base64.b64encode(buffered_img.getvalue()).decode("utf-8")

    preds = model_predict(img, model)
    result = "Chien" if preds[0][0] < 0.5 else "Chat"

    with mlflow.start_run(nested=True):
        mlflow.log_metric("Confident_score", float(preds[0][0]))
        mlflow.log_param("predicted_result", result)
        mlflow.set_tag("Prediction Info", "Initial prediction for user-uploaded image")
    mlflow.end_run()

    return render_template(
        "result.html",
        result=result,
        user_input=f.filename,
        image_base64_front=base64_img,
    )


@app.route("/validate_prediction", methods=["POST"])
def validate_prediction():
    user_input = request.form.get("user_input")
    model_output = request.form.get("model_output")
    if not user_input or not model_output:
        flash("Erreur : Données manquantes.", "error")
        return redirect(url_for("home"))

    feedback = "validated"
    timestamp = datetime.now()

    with mlflow.start_run():
        mlflow.log_param("user_input", user_input)
        mlflow.log_param("model_output", model_output)
        mlflow.log_param("feedback", feedback)
        mlflow.log_param("timestamp", timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        flash("Prédiction validée avec succès!", "success")
    mlflow.end_run()

    new_feedback = Feedback(
        user_input=user_input,
        model_output=model_output,
        feedback=feedback,
        timestamp=timestamp,
    )
    db.session.add(new_feedback)
    db.session.commit()

    return redirect(url_for("home"))

@app.route("/mlflow_dashboard")
@login_required
def mlflow_dashboard():
    return redirect("http://localhost:5011", code=302)


@app.route("/reject_prediction", methods=["POST"])
def reject_prediction():
    user_input = request.form.get("user_input")
    model_output = request.form.get("model_output")
    real_label = request.form.get("real_label")
    image_base64 = request.form.get("image_base64")

    if not user_input or not model_output or not real_label or not image_base64:
        flash("Erreur : Données manquantes.", "error")
        return redirect(url_for("home"))

    feedback = "rejected"
    timestamp = datetime.now()
    file_name = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{user_input}"
    image_path = f"data/retraining/{real_label}/{file_name}.jpg"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    img_data = base64.b64decode(image_base64)
    with open(image_path, "wb") as f:
        f.write(img_data)

    with mlflow.start_run():
        mlflow.log_param("user_input", user_input)
        mlflow.log_param("model_output", model_output)
        mlflow.log_param("feedback", feedback)
        mlflow.log_param("real_label", real_label)
        mlflow.log_param("timestamp", timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        mlflow.log_artifact(image_path, artifact_path="rejected_images")
        flash("Prédiction rejetée enregistrée.", "danger")
    mlflow.end_run()

    new_feedback = Feedback(
        user_input=user_input,
        model_output=model_output,
        feedback=feedback,
        timestamp=timestamp,
    )
    db.session.add(new_feedback)
    db.session.commit()

    if should_retrain_model():
        retrain_model()
        flash("Modèle réentraîné avec succès!", "success")

    return redirect(url_for("home"))


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
