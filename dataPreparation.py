from datab import db, Feedback
import pandas as pd

def fetch_feedback_data():
    """ Récupère les données de feedback de la base de données. """
    feedback_records = Feedback.query.all()
    data = pd.DataFrame([{
        'user_input': f.user_input,
        'model_output': f.model_output,
        'feedback': f.feedback,
        'timestamp': f.timestamp
    } for f in feedback_records])
    print(data)
    return data



# def prepare_data_for_training(data):
#     """ Nettoie et prépare les données pour le ré-entraînement. """
#     # Ici, vous pouvez ajouter le nettoyage spécifique requis pour votre modèle.
#     # Par exemple, convertir les timestamps en features saisonnières, ou ré-étiqueter les données selon le feedback.
    
#     # Exemple de ré-étiquetage basé sur le feedback
#     data.loc[data['feedback'] == 'rejected', 'correct_label'] = data['user_input']  # Supposition simpliste
#     data.loc[data['feedback'] != 'rejected', 'correct_label'] = data['model_output']
    
#     return data[['user_input', 'correct_label']]

from keras.preprocessing import image
import numpy as np

#Formatage des Étiquettes pour la Classification
def format_labels(labels, class_mapping):
    # class_mapping est un dictionnaire qui mappe les noms de classe à des indices
    # par exemple, {'cat': 0, 'dog': 1}
    formatted_labels = [class_mapping[label] for label in labels]
    print(formatted_labels)
    return formatted_labels

#Utilisation de Keras pour le Formatage des Étiquettes
from keras.utils import to_categorical

def prepare_labels(labels, num_classes):
    return to_categorical(labels, num_classes=num_classes)

#Intégration dans le Pipeline de Préparation des Données
def prepare_images_and_labels(data_entries, image_size, class_mapping):
    images = []
    labels = []
    
    for user_input, model_output, feedback in data_entries:
        img = image.load_img(user_input, target_size=(image_size, image_size))
        img_array = image.img_to_array(img)
        images.append(img_array)
        
        # Déterminer la bonne étiquette
        correct_label = user_input.split('/')[-2] if feedback == 'rejected' else model_output.split('/')[-2]
        labels.append(correct_label)

    images = np.array(images) / 255.0
    formatted_labels = format_labels(labels, class_mapping)
    print(images)
    print(labels)
    return images, formatted_labels


import matplotlib.pyplot as plt

def plot_sample_images(images, labels, num_images=5):
    """
    Affiche un échantillon d'images avec des étiquettes.
    :param images: tableau numpy d'images.
    :param labels: liste des étiquettes associées aux images.
    :param num_images: nombre d'images à afficher.
    """
    fig, axes = plt.subplots(1, num_images, figsize=(20, 4))
    for i, ax in enumerate(axes):
        # Assurez-vous que l'index ne dépasse pas la longueur de votre dataset
        index = i if i < len(images) else len(images) - 1
        ax.imshow(images[index].astype('uint8'))
        ax.title.set_text(f'Label: {labels[index]}')
        ax.axis('off')
    plt.show()

# Supposons que vous ayez déjà vos images et étiquettes préparées
# images, labels = prepare_images_and_labels(data, IMAGE_SIZE, class_mapping)
# plot_sample_images(images, labels, num_images=5)


from flask import Flask
from datab import db, Feedback
import pandas as pd

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///path_to_your_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def fetch_feedback_data():
    with app.app_context():  # Créer un contexte d'application ici
        feedback_records = Feedback.query.all()
        data = pd.DataFrame([{
            'user_input': f.user_input,
            'model_output': f.model_output,
            'feedback': f.feedback,
            'timestamp': f.timestamp
        } for f in feedback_records])
        print(data)
        return data

if __name__ == '__main__':
    data = fetch_feedback_data()
