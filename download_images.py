import requests
import os

def download_from_api(api_url, folder, label, total=10):
    os.makedirs(folder, exist_ok=True)
    for i in range(total):
        try:
            if label == "chat":
                response = requests.get(api_url)
                img_url = response.json()[0]['url']
            else:
                response = requests.get(api_url)
                img_url = response.json()['message']

            img_data = requests.get(img_url).content
            with open(os.path.join(folder, f"{i}.jpg"), 'wb') as handler:
                handler.write(img_data)
            print(f"✅ {label.capitalize()} {i}.jpg téléchargée.")
        except Exception as e:
            print(f"❌ Erreur téléchargement {label} {i} :", e)

# TheCatAPI → 1 image par appel
cat_api_url = "https://api.thecatapi.com/v1/images/search"

# Dog CEO → 1 image par appel
dog_api_url = "https://dog.ceo/api/breeds/image/random"

download_from_api(cat_api_url, "data/images/chat", "chat", total=10)
download_from_api(dog_api_url, "data/images/chien", "chien", total=10)

