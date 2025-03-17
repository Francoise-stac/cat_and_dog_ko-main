import mlflow.keras
import numpy as np
from keras.preprocessing import image
import os

# Configurer l'URI de suivi de MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# Charger la version 19 du modèle enregistré dans MLflow
model_uri = "models:/Model_for_User_Feedback/19"
model = mlflow.keras.load_model(model_uri)

# Fonction pour prétraiter l'image
def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension pour le batch
    img_array = img_array / 255.0  # Normaliser les valeurs des pixels
    return img_array

# Effectuer une prédiction
def predict_with_new_model(img_path):
    img_preprocessed = preprocess_image(img_path)
    prediction = model.predict(img_preprocessed)
    return prediction

# Lister les fichiers dans le répertoire pour vérifier
dir_path = r"C:\Users\Utilisateur\Documents\29_Debbugage\data\retraining\chien"
files_in_dir = os.listdir(dir_path)

print(f"Fichiers trouvés dans le répertoire '{dir_path}':")
for f in files_in_dir:
    print(f)
import os

# Dossier contenant les fichiers à renommer
directory = r"C:\Users\Utilisateur\Documents\29_Debbugage\data\retraining\chien"

# Parcourir tous les fichiers dans le répertoire
for filename in os.listdir(directory):
    # Si le fichier a le double suffixe .jfif.jpg, on le renomme
    if filename.endswith(".jfif.jpg"):
        new_filename = filename.replace(".jfif.jpg", ".jfif")  # Enlever le .jpg supplémentaire
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
        print(f"Renommé : {filename} -> {new_filename}")

# Chemin vers l'image pour laquelle tu veux faire une prédiction
img_path = r"C:\Users\Utilisateur\Documents\29_Debbugage\data\retraining\chien\image1.jpg"  # Remplace par le chemin de ton image

if os.path.exists(img_path):
    result = predict_with_new_model(img_path)
    # Afficher les résultats
    if result[0][0] < 0.5:
        print(f"La prédiction pour l'image '{img_path}' est : Chien")
    else:
        print(f"La prédiction pour l'image '{img_path}' est : Chat")
else:
    print(f"Le chemin de l'image '{img_path}' est invalide ou le fichier n'existe pas.")

# Exemple d'utilisation
if __name__ == "__main__":
    # Chemin vers l'image pour laquelle tu veux faire une prédiction
    img_path = r"C:\Users\Utilisateur\Documents\29_Debbugage\data\retraining\chien\20241003_223518_chein.jfif"  # Remplace par le chemin de ton image

    if os.path.exists(img_path):
        result = predict_with_new_model(img_path)

        # Afficher les résultats
        if result[0][0] < 0.5:
            print(f"La prédiction pour l'image '{img_path}' est : Chien")
        else:
            print(f"La prédiction pour l'image '{img_path}' est : Chat")
    else:
        print(f"Le chemin de l'image '{img_path}' est invalide ou le fichier n'existe pas.")
