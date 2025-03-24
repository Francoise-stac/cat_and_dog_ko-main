# Supprimer le conteneur s'il existe déjà
docker rm -f catdog_container

# Rebuild l’image Docker
docker build -t catdog_image .

# Lancer le conteneur avec montage correct
docker run -d `
  -p 5000:5000 `
  -p 5001:5001 `
  -v "${PWD}/instance:/app/instance" `
  -v "${PWD}/mlruns:/app/mlruns" `
  -v "${PWD}/data/retraining:/app/data/retraining" `
  --name catdog_container catdog_image
