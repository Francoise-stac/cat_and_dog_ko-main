import os
import logging
from flask import Flask, request, render_template, redirect, url_for, flash, session
from functools import wraps
import numpy as np
from keras.models import load_model
from io import BytesIO
from PIL import Image
import base64
import mlflow
from datetime import datetime
from keras import backend as K
import keras
from models import db, Feedback, User
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from keras.preprocessing import image
from glob import glob
import tensorflow as tf
from dotenv import load_dotenv
import traceback
from mlflow.exceptions import MlflowException
from flask_monitoringdashboard import bind, config
from mlflow_config import get_or_create_experiment
import flask_monitoringdashboard as dashboard


# Important pour éviter que .dashboard.cfg interfère
config.init_from(file=False)
config.security_token = None
config.basicAuth = False



experiment_id = get_or_create_experiment()

instance_path = os.path.abspath("instance")

# 🧠 Créer l'application après la config
app = Flask(__name__)


bind(app)

# load_dotenv()


app.secret_key = "une_clé_secrète_pour_session_et_flash"
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'votre_clé_secrète_par_défaut')

# Journalisation
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("🚀 Application démarrée.")

SEED = 42
IMAGE_SIZE = 128
MAX_IMG = 4000

TESTING = os.environ.get('TESTING', 'False') == 'True'

if not TESTING:
    mlflow.set_tracking_uri("http://127.0.0.1:5001")  #mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5001

    mlflow.set_experiment("feedback_experiment_clean")

# ARTIFACT_ROOT = os.path.abspath("mlruns_artifacts")
ARTIFACT_ROOT = "file:///app/artifacts" 
EXPERIMENT_NAME = "feedback_experiment_clean"

try:
    experiment_id = mlflow.create_experiment(
        name=EXPERIMENT_NAME,
        artifact_location=ARTIFACT_ROOT
    )
    print(f"✅ Expérience créée : {EXPERIMENT_NAME}")
except MlflowException:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
    print(f"⚠️ Expérience déjà existante : {EXPERIMENT_NAME}")

mlflow.set_experiment(EXPERIMENT_NAME)



BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TESTING = os.environ.get('TESTING', 'False') == 'True'

if TESTING:
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
else:
    if os.environ.get("DOCKER", "false").lower() == "true":
        # En Docker (tu peux aussi utiliser un autre indicateur)
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////app/data/mlflow.db"
    else:
        # En local
        db_path = os.path.join(BASE_DIR, "data", "mlflow.db")
        app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"




app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
migrate = Migrate(app, db)

with app.app_context():
    db.create_all()

MODEL_PATH = "models/model.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le fichier modèle n'existe pas : {MODEL_PATH}")

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


def retrain_model():
    logging.info("🔁 Début du réentraînement du modèle...")
    IMAGE_SIZE = 128
    data_dir = 'data/retraining'

    # Initialisation des listes pour les images et les labels
    X = []
    y = []

    for index_animal, animal in enumerate(['chat', 'chien']):
        for img_path in glob(os.path.join(data_dir, animal, "*.jpg")):
            try:
                # Chargement de l'image
                img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
                # Conversion en tableau Numpy
                img_array = image.img_to_array(img)
                # Ajout de l'image et du label à la liste
                X.append(img_array)
                y.append(index_animal)
            except Exception as e:
                # print(f"Erreur lors du chargement de l'image {img_path}: {e}")
                logging.error(f"Erreur lors du chargement de l'image {img_path}: {e}")

    # Conversion des listes en tableaux Numpy
    X = np.array(X)  # Forme (n_samples, 128, 128, 3)
    y = np.array(y)  # Forme (n_samples,)

    # Normalisation des images (valeurs des pixels entre 0 et 1)
    X = X / 255.0

    # Encodage des labels en one-hot encoding
    y = keras.utils.to_categorical(y, num_classes=2)

    # Vérification des formes des données
    print(f"Forme de X : {X.shape}")
    print(f"Forme de y : {y.shape}")

    # Charger l'ancien modèle
    model = load_model(MODEL_PATH)

    logging.info(f"🔁 Réentraînement déclenché - {len(X)} images utilisées (chat : {y.tolist().count(0)}, chien : {y.tolist().count(1)})")

    # Réentraîner le modèle avec les nouvelles données
    model.fit(X, y, epochs=5, batch_size=32)

    # Sauvegarder le modèle mis à jour
    model.save(MODEL_PATH)
    print(f"Modèle réentraîné et sauvegardé dans : {MODEL_PATH}")
    logging.info("✅ Modèle réentraîné et sauvegardé avec succès.")


    # Créer une nouvelle version du modèle dans MLflow
    # with mlflow.start_run():
    with mlflow.start_run(experiment_id=experiment_id):

        mlflow.keras.log_model(model, "model", registered_model_name="Model_for_User_Feedback")
        print("Modèle réentraîné sauvegardé dans MLflow.")
    mlflow.end_run()




# def login_required(f):
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#         if 'user_id' not in session:
#             flash('Veuillez vous connecter pour accéder à cette page.', 'error')
#             return redirect(url_for('login'))
#         return f(*args, **kwargs)
#     return decorated_function

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Ne bloque pas les routes du dashboard
        if request.path.startswith('/dashboard'):
            return f(*args, **kwargs)

        if 'user_id' not in session:
            flash('Veuillez vous connecter pour accéder à cette page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


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
        user.set_password(password)  # Hacher le mot de passe

        # Ajouter l'utilisateur à la base de données
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


@app.route("/reject_prediction", methods=["POST"])
def reject_prediction():
    print("🔄 Début de reject_prediction")

    user_input = request.form.get("user_input")
    model_output = request.form.get("model_output")
    real_label = request.form.get("real_label")
    image_base64 = request.form.get("image_base64")

    print("🧾 Champs reçus:")
    print(" - user_input:", user_input)
    print(" - model_output:", model_output)
    print(" - real_label:", real_label)
    print(" - image_base64 present:", image_base64 is not None)

    if not user_input or not model_output or not real_label or not image_base64:
        flash("Erreur : Données manquantes.", "error")
        return redirect(url_for("home"))

    feedback = "rejected"
    timestamp = datetime.now()
    file_name = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{user_input}.jpg"
    image_path = os.path.join("data", "retraining", real_label, file_name)
    # artifact_dir = os.path.join(os.getcwd(), "artifacts", "rejected_images")
    artifact_dir = os.path.join("artifacts", "rejected_images")

    logging.warning(f"❌ Prédiction rejetée par l'utilisateur - Image : {user_input}, Prédiction : {model_output}, Attendu : {real_label}")


    try:
        print("📁 Création du dossier d’image:", os.path.dirname(image_path))
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        print("📁 Création du dossier pour artefacts MLflow:", artifact_dir)
        os.makedirs(artifact_dir, exist_ok=True)

        print("💾 Sauvegarde de l’image...")
        img_data = base64.b64decode(image_base64)
        with open(image_path, "wb") as f:
            f.write(img_data)
        print("✅ Image sauvegardée:", image_path)

        print("🔗 Connexion à MLflow...")

        # with mlflow.start_run():
        with mlflow.start_run(experiment_id=experiment_id):

            mlflow.log_param("user_input", user_input)
            mlflow.log_param("model_output", model_output)
            mlflow.log_param("feedback", feedback)
            mlflow.log_param("real_label", real_label)
            mlflow.log_param("timestamp", timestamp.strftime("%Y-%m-%d %H:%M:%S"))
            print("📦 Enregistrement de l’image dans MLflow...")
            print("📸 image_path utilisé :", image_path)

            mlflow.log_artifact(image_path, artifact_path="rejected_images")



            flash("Prédiction rejetée enregistrée.", "danger")
            print("✅ Image loggée dans MLflow.")

    except Exception as e:
        print("❌ Erreur lors de l’enregistrement du rejet :", str(e))
        logging.error(f"Erreur lors du traitement de rejet : {str(e)}")

        traceback.print_exc() 
        flash(f"Erreur MLflow : {e}", "error")
        return redirect(url_for("home"))

    print("📊 Enregistrement dans la base de données...")
    new_feedback = Feedback(
        user_input=user_input,
        model_output=model_output,
        feedback=feedback,
        timestamp=timestamp,
    )
    db.session.add(new_feedback)
    db.session.commit()
    print("✅ Feedback sauvegardé.")

    if should_retrain_model():
        print("🔁 Seuil atteint, début du réentraînement.")
        retrain_model()
        flash("Modèle réentraîné avec succès!", "success")

    print("🔚 Fin de reject_prediction")
    return redirect(url_for("home"))



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

    logging.info(f"🧠 Prédiction effectuée sur : {f.filename}, Résultat : {result}")

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

    # with mlflow.start_run():
    with mlflow.start_run(experiment_id=experiment_id):

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

    logging.info(f"✅ Prédiction validée : {user_input}, classe prédite : {model_output}")
    
    return redirect(url_for("home"))

def register_user(data):
    return {"success": True, "user_id": 123, "message": "Inscription réussie"}

def authenticate_user(credentials):
    return {"success": True, "token": "jwt_token_xyz", "user_id": 123}

def save_prediction(user_id, prediction):
    return {"status": "success", "record_id": 789}

def get_user_predictions(user_id):
    return [
        {"id": 1, "result": "cat", "timestamp": "2023-01-01T12:00:00"},
        {"id": 2, "result": "dog", "timestamp": "2023-01-02T14:30:00"}
    ]

def predict(image_data):
    return {"class": "cat", "probability": 0.87, "processing_time": 0.156}



def save_result(data):
    return {"status": "success", "record_id": 999}
def retrieve_history(user_id):
    return [
        {"id": 1, "result": "cat", "timestamp": "2023-01-01T12:00:00"},
        {"id": 2, "result": "dog", "timestamp": "2023-01-02T14:30:00"}
    ]


# Facultatif si tu veux expliciter ce que le module expose
__all__ = [
    "register_user",
    "authenticate_user",
    "save_prediction",
    "get_user_predictions",
    "predict"
]



if __name__ == "__main__":
    # app.run(debug=True)
    logging.info("🟢 Serveur Flask lancé (port 5000)")

    app.run(host="0.0.0.0", port=5000, debug=True)


