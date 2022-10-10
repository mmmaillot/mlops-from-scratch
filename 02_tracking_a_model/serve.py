from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc

app = FastAPI()


class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class Label(BaseModel):
    label: int


@app.on_event("startup")
async def startup_event():
    global model
    model_name = "sk-learn-dummy-model"
    model_stage = "Production"

    mlflow.set_tracking_uri("http://localhost:5001")
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_stage}"
    )


@app.post("/label", response_model=Label)
async def label(iris: Iris):
    iris_data = [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]]
    label = model.predict(iris_data)
    return Label(label=label[0])
