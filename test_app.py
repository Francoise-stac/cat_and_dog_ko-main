import pytest
from app import app, db
from models import User, Feedback

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client
        with app.app_context():
            db.drop_all()

def test_home(client):
    """Teste si la route d'accueil '/' nécessite une connexion"""
    response = client.get('/')
    assert response.status_code == 302  # Redirection vers la page de login
    assert b"login" in response.headers['Location']

def test_register(client):
    """Teste l'inscription d'un nouvel utilisateur"""
    response = client.post('/register', data={
        'username': 'testuser',
        'email': 'testuser@example.com',
        'password': 'password123'
    }, follow_redirects=True)
    assert response.status_code == 200
    assert "Inscription réussie".encode('utf-8') in response.data

def test_login(client):
    """Teste la connexion d'un utilisateur existant"""
    # Créer un utilisateur
    with app.app_context():
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

def test_prediction(client):
    """Teste la prédiction avec une image fictive"""
    # Simuler une connexion
    with app.app_context():
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

def test_validate_prediction(client):
    """Teste la validation d'une prédiction"""
    with app.app_context():
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

def test_reject_prediction(client):
    """Teste le rejet d'une prédiction"""
    with app.app_context():
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