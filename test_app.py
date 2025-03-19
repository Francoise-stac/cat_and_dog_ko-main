import os
import pytest
from unittest.mock import patch
from io import BytesIO
from models import User, db

# Configuration pour les tests
os.environ['TESTING'] = 'True'

@pytest.fixture
def flask_app():  # Renommé de 'app' pour éviter les conflits
    from app import app
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    return app

@pytest.fixture
def client(flask_app):  # Mise à jour pour utiliser flask_app
    return flask_app.test_client()

@pytest.fixture
def init_database(flask_app):  # Mise à jour pour utiliser flask_app
    with flask_app.app_context():
        db.create_all()
        yield db
        db.session.remove()
        db.drop_all()

@pytest.fixture(autouse=True)
def mock_mlflow():
    with patch('mlflow.set_tracking_uri'), \
         patch('mlflow.set_experiment'), \
         patch('mlflow.start_run', return_value=type('obj', (object,), {'__enter__': lambda x: None, '__exit__': lambda x, *args: None})), \
         patch('mlflow.end_run'), \
         patch('mlflow.log_metric'), \
         patch('mlflow.log_param'), \
         patch('mlflow.log_artifact'), \
         patch('mlflow.set_tag'), \
         patch('mlflow.keras.log_model'):
        yield

@pytest.fixture(autouse=True)
def mock_model():
    with patch('app.model_predict', return_value=[[0.7]]), \
         patch('app.load_model'), \
         patch('app.should_retrain_model', return_value=False):
        yield

def test_home(client):
    """Teste si la route d'accueil '/' nécessite une connexion"""
    response = client.get('/')
    assert response.status_code == 302
    assert 'login' in response.headers['Location']  # Correction du type string vs bytes

def test_register(client, init_database):
    """Teste l'inscription d'un nouvel utilisateur"""
    response = client.post('/register', data={
        'username': 'testuser',
        'email': 'testuser@example.com',
        'password': 'password123'
    }, follow_redirects=True)
    assert response.status_code == 200
    assert "Inscription réussie".encode('utf-8') in response.data

def test_login(client, init_database, flask_app):  # Ajout de flask_app
    """Teste la connexion d'un utilisateur existant"""
    with flask_app.app_context():  # Utilisation de flask_app au lieu de app
        user = User(username='testuser', email='testuser@example.com')
        user.set_password('password123')
        db.session.add(user)
        db.session.commit()

    # Tester la connexion
    response = client.post('/login', data={
        'username': 'testuser',
        'password': 'password123'
    }, follow_redirects=True)
    assert response.status_code == 200
    assert "Connexion réussie".encode('utf-8') in response.data

def test_prediction(client, init_database, flask_app):  # Ajout de flask_app
    """Teste la prédiction avec une image fictive"""
    # Simuler une connexion
    with flask_app.app_context():  # Utilisation de flask_app au lieu de app
        user = User(username='testuser', email='testuser@example.com')
        user.set_password('password123')
        db.session.add(user)
        db.session.commit()

    client.post('/login', data={
        'username': 'testuser',
        'password': 'password123'
    })

    # Tester la prédiction
    data = {
        'file': (BytesIO(b"fake image data"), "test.jpg")
    }
    response = client.post('/result', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert b"Chat" in response.data or b"Chien" in response.data

def test_validate_prediction(client, init_database, flask_app):  # Ajout de flask_app
    """Teste la validation d'une prédiction"""
    with flask_app.app_context():  # Utilisation de flask_app au lieu de app
        user = User(username='testuser', email='testuser@example.com')
        user.set_password('password123')
        db.session.add(user)
        db.session.commit()

    client.post('/login', data={
        'username': 'testuser',
        'password': 'password123'
    })

    response = client.post('/validate_prediction', data={
        'user_input': 'test.jpg',
        'model_output': 'Chat'
    }, follow_redirects=True)
    assert response.status_code == 200
    assert "Prédiction validée".encode('utf-8') in response.data

def test_reject_prediction(client, init_database, flask_app):  # Ajout de flask_app
    """Teste le rejet d'une prédiction"""
    with flask_app.app_context():  # Utilisation de flask_app au lieu de app
        user = User(username='testuser', email='testuser@example.com')
        user.set_password('password123')
        db.session.add(user)
        db.session.commit()

    client.post('/login', data={
        'username': 'testuser',
        'password': 'password123'
    })

    response = client.post('/reject_prediction', data={
        'user_input': 'test.jpg',
        'model_output': 'Chat',
        'real_label': 'chien',
        'image_base64': 'fake_base64_data'
    }, follow_redirects=True)
    assert response.status_code == 200
    assert "Prédiction rejetée".encode('utf-8') in response.data

def test_db_configuration(flask_app):  # Mise à jour pour utiliser flask_app
    """Vérifie que la configuration de la base de données est correcte pour les tests"""
    assert flask_app.config['SQLALCHEMY_DATABASE_URI'] == 'sqlite:///:memory:'
    assert flask_app.config['TESTING'] is True

def test_db_operations(init_database, client):
    """Teste les opérations de base de données"""
    from app import User
    
    with client.application.app_context():
        # Créer un utilisateur test
        user = User(username='testuser', email='test@test.com')
        user.set_password('password123')
        init_database.session.add(user)
        init_database.session.commit()
        
        # Vérifier que l'utilisateur est bien créé
        assert User.query.filter_by(username='testuser').first() is not None