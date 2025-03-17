from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_input = db.Column(db.String(255), nullable=False)
    model_output = db.Column(db.String(255), nullable=False)
    feedback = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
