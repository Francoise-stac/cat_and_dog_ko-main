import pytest
from app import app

@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

def test_home(client):
    """Teste si la route d'accueil '/' fonctionne"""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Bienvenue" in response.data  # Vérifier que la page contient du tex

def test_classification(client):
    response = client.post('/predict', json={"image_path": "test.jpg"})
    assert response.status_code == 200
    assert "prediction" in response.json

def test_prediction(client):
    """Teste si la prédiction fonctionne avec une image fictive"""
    data = {'file': (open("test.jpg", "rb"), "test.jpg")}
    response = client.post('/result', data=data)
    assert response.status_code == 200
    assert b"prediction" in response.data  # Vérifier que la réponse contient une prédiction