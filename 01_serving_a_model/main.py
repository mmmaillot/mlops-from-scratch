from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from fastapi import FastAPI
from pydantic import BaseModel

iris_dataset = load_iris()
X, y = iris_dataset["data"], iris_dataset["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

dummy_classifier = DummyClassifier(strategy="constant", constant=y[0])
dummy_classifier.fit(X_train, y_train)

app = FastAPI()


class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class Label(BaseModel):
    label: int


@app.post("/label", response_model=Label)
async def label(iris: Iris):
    iris_data = [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]]
    label = dummy_classifier.predict(iris_data)
    return Label(label=label[0])
