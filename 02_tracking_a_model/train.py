from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

iris_dataset = load_iris()
X, y = iris_dataset["data"], iris_dataset["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
dummy_classifier = DummyClassifier(strategy="constant", constant=y[0])
dummy_classifier.fit(X_train, y_train)

mlflow.set_tracking_uri("http://localhost:5001")

with mlflow.start_run(run_name="EXAMPLE_RUN") as run:
    # Log the sklearn model and register as version 1
    mlflow.sklearn.log_model(
        sk_model=dummy_classifier,
        artifact_path="sklearn-model",
        registered_model_name="sk-learn-dummy-model"
    )

client = MlflowClient()
client.transition_model_version_stage(
    name="sk-learn-dummy-model",
    version=1,
    stage="Production"
)
