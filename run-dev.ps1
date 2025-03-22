# Arrêter et supprimer le container s'il existe
docker stop catdog_dev 2>$null
docker rm catdog_dev 2>$null

# Lancer avec tout le dossier monté
docker run -it `
  -v "C:\Users\Francy\Documents\cat_and_dog_ko-main:/app" `
  -p 5010:5000 `
  -p 5011:5001 `
  --name catdog_dev `
  cat-dog-app
