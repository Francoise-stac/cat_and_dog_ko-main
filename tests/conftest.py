import pytest
import os
from PIL import Image
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

@pytest.fixture(scope="session")
def test_image():
    """Créer une image de test"""
    img_path = "tests/test_data/cat.jpg"
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    
    # Créer une image test
    img = Image.new('RGB', (128, 128), color='red')
    img.save(img_path)
    return img_path

@pytest.fixture(scope="session")
def chrome_options():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    return options

@pytest.fixture(scope="session")
def browser(chrome_options):
    if os.getenv('GITHUB_ACTIONS'):
        # Configuration pour GitHub Actions
        options = chrome_options
        options.add_argument('--disable-gpu')
        driver = webdriver.Remote(
            command_executor=os.getenv('SELENIUM_HUB_URL', 'http://localhost:4444/wd/hub'),
            options=options
        )
    else:
        # Configuration locale
        driver = webdriver.Chrome(options=chrome_options)
    
    driver.implicitly_wait(10)
    yield driver
    driver.quit()