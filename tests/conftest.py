import pytest
import os
from PIL import Image
import numpy as np

@pytest.fixture(scope="session")
def test_image():
    """Créer une image de test"""
    img_path = "tests/test_data/cat.jpg"
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    
    # Créer une image test
    img = Image.new('RGB', (128, 128), color='red')
    img.save(img_path)
    return img_path