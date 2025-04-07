import mlflow
import os
from mlflow.exceptions import MlflowException

EXPERIMENT_NAME = "feedback_experiment_clean"
ARTIFACT_ROOT = os.path.abspath("mlruns_artifacts").replace(os.sep, '/')

mlflow.set_tracking_uri("http://127.0.0.1:5001")


def get_or_create_experiment():
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        try:
            experiment_id = mlflow.create_experiment(
                name=EXPERIMENT_NAME,
                artifact_location=f"file:///{ARTIFACT_ROOT}"
            )
            print(f"✅ Expérience MLflow créée : {EXPERIMENT_NAME}")
        except MlflowException as e:
            print("❌ Erreur lors de la création de l'expérience :", e)
            raise
    else:
        experiment_id = experiment.experiment_id
        print(f"⚠️ Expérience déjà existante : {EXPERIMENT_NAME}")

    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id
