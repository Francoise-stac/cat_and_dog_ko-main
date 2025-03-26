import os
import logging
from logging.handlers import RotatingFileHandler, SMTPHandler

def setup_logging(app):
    # Crée le répertoire de logs si nécessaire
    if not os.path.exists('logs'):
        os.makedirs('logs')

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s [in %(pathname)s:%(lineno)d]'
    )

    # Log dans un fichier avec rotation
    file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)

    # Handler pour envoi d'e-mail en cas d'erreur critique (si MAIL_SERVER est défini dans les variables d'environnement)
    if os.getenv('MAIL_SERVER'):
        mail_handler = SMTPHandler(
            mailhost=(os.getenv('MAIL_SERVER'), int(os.getenv('MAIL_PORT', 587))),
            fromaddr=os.getenv('MAIL_USERNAME'),
            toaddrs=['admin@example.com'],  # Remplacez par les e-mails des administrateurs
            subject='[ALERTE] Problème sur l’App Flask',
            credentials=(os.getenv('MAIL_USERNAME'), os.getenv('MAIL_PASSWORD')),
            secure=()
        )
        mail_handler.setLevel(logging.ERROR)
        mail_handler.setFormatter(formatter)
        app.logger.addHandler(mail_handler)

    app.logger.setLevel(logging.INFO)
    app.logger.info("L'application Flask a démarré !")