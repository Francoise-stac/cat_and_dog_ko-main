from pymongo import MongoClient
from bson import Binary
import os

client = MongoClient("mongodb://localhost:27017/")
db = client.chat_chien_db
collection = db.images

def insert_images(folder, label):
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            with open(os.path.join(folder, filename), "rb") as f:
                img_data = f.read()
                doc = {
                    "label": label,
                    "filename": filename,
                    "image": Binary(img_data)
                }
                collection.insert_one(doc)
                print(f"✅ Image {filename} enregistrée dans MongoDB avec label {label}")

insert_images("data/images/chat", "chat")
insert_images("data/images/chien", "chien")
