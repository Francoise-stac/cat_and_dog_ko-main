# Arrêter et supprimer l’ancien container
docker stop catdog_app 2>$null
docker rm catdog_app 2>$null

# Rebuild complet de l'image
docker build -t cat-dog-app .

# Lancer avec uniquement le volume des images rejetées
docker run -it `
  -v "C:\Users\Francy\Documents\cat_and_dog_ko-main\data\retraining:/app/data/retraining" `
  -p 5010:5000 `
  -p 5011:5001 `
  --name catdog_app `
  cat-dog-app
