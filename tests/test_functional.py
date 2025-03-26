import os
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

@pytest.fixture(scope="module")
def browser():
    """Fixture pour créer et configurer le navigateur"""
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')  # Mode sans interface graphique
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), 
                            options=chrome_options)
    driver.implicitly_wait(10)
    yield driver
    driver.quit()

@pytest.fixture(scope="module")
def test_app():
    """Fixture pour démarrer l'application Flask en mode test"""
    from app import app
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with app.test_client() as client:
        with app.app_context():
            from models import db
            db.create_all()
            yield app
            db.session.remove()
            db.drop_all()

# def test_home_page_title(browser, test_app):
#     """Test que le titre de la page d'accueil est correct"""
#     browser.get('http://localhost:5000')
#     assert "Cat & Dog Classifier" in browser.title

# def test_user_registration_flow(browser, test_app):
#     """Test du processus complet d'inscription"""
#     browser.get('http://localhost:5000/register')
    
#     # Remplir le formulaire d'inscription
#     username_input = browser.find_element(By.NAME, "username")
#     email_input = browser.find_element(By.NAME, "email")
#     password_input = browser.find_element(By.NAME, "password")
    
#     username_input.send_keys("testuser")
#     email_input.send_keys("test@example.com")
#     password_input.send_keys("password123")
    
#     # Soumettre le formulaire
#     password_input.submit()
    
#     # Vérifier le message de succès
#     success_message = WebDriverWait(browser, 10).until(
#         EC.presence_of_element_located((By.CLASS_NAME, "alert-success"))
#     )
#     assert "Inscription réussie" in success_message.text

# def test_login_flow(browser, test_app):
#     """Test du processus de connexion"""
#     browser.get('http://localhost:5000/login')
    
#     # Remplir le formulaire de connexion
#     username_input = browser.find_element(By.NAME, "username")
#     password_input = browser.find_element(By.NAME, "password")
    
#     username_input.send_keys("testuser")
#     password_input.send_keys("password123")
    
#     # Soumettre le formulaire
#     password_input.submit()
    
#     # Vérifier la redirection vers la page d'accueil
#     WebDriverWait(browser, 10).until(
#         EC.presence_of_element_located((By.ID, "upload-form"))
#     )
#     assert browser.current_url == "http://localhost:5000/"

# def test_image_upload_flow(browser, test_app):
    """Test du processus de téléchargement et prédiction d'image"""
    browser.get('http://localhost:5000')
    
    # S'assurer d'être connecté
    if "login" in browser.current_url:
        test_login_flow(browser, test_app)
    
    # Télécharger une image
    file_input = browser.find_element(By.NAME, "file")
    file_input.send_keys(os.path.abspath("tests/test_data/cat.jpg"))
    
    # Soumettre le formulaire
    submit_button = browser.find_element(By.ID, "submit-button")
    submit_button.click()
    
    # Vérifier le résultat
    result = WebDriverWait(browser, 10).until(
        EC.presence_of_element_located((By.ID, "prediction-result"))
    )
    assert any(animal in result.text for animal in ["Chat", "Chien"])