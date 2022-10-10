---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Tracking your model

First things first, we will now separate the training of the model, in a file called `train.py` from the serving of the model, in a file called `serve.py`. That is a good first step. But that won't be enough.

Because, what are we going to do once that model has been trained ?

## Saving your models

Our first model was the simplest model you could ever imagine : simply answering the same label every time. It takes no time to train. That's why we did not even bother to save it.

If your model takes minutes, hours or even days to train, you can't train it at the startup of your container. You are going to train it and save it. Then at the startup of your container, you will load it. Loading is supposedly faster than training.

```python
# This should be in a file called train.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

iris_dataset =  load_iris()
X,  y = iris_dataset["data"], iris_dataset["target"]
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
dummy_classifier = DummyClassifier(strategy="constant", constant=y[0])
dummy_classifier.fit(X_train,y_train)
```

Now we want to save it. We are going to use the built-in solution for Python persistence : `pickle`

```python
import pickle

file_path = "./dumb_model.pkl"

with open(file_path, "wb") as f:
    pickle.dump(dummy_classifier , f)
```

Are we sure this is the same model ? It might be altered from the persistence process. Let's check that.

```python
with open(file_path, "rb") as f:
    loaded_model = pickle.load(f)
    
assert (loaded_model.predict(X_test[0:10]) == dummy_classifier.predict(X_test[0:10])).all()
```

Yay ! It seems to produce the same results as the one before pickling.
So now we can put this code into the `serve.py` file, that looks almost the same as the `main.py`.

This step is very important : we have introduced a decoupling between the training and the serving. I personally think that this is the most important thing to do, and that everything is straightforward from there.

## Organizing your models

Before we dive any further, let's think about everything we will need for the next step.

* Models need to be stored and easily retrievable for your teammates, and for your business applications, cloud apps included
* Models should be unpickled in an environment similar to the one they have been pickled in
* When you create a new model, you should be able to compare it to the one in production
* It should be easy to know **which model is in production**

Basically, what you could do on your own, is storing your models in a cloud bucket, such as [Amazon S3](https://aws.amazon.com/en/s3/), [Google Cloud Storage](https://cloud.google.com/storage/docs/creating-buckets), [Azure Blob Storage](https://azure.microsoft.com/fr-fr/products/storage/blobs/) or whatever.

You could also log all the metada related to the model, the environment from which it has been pickled, and the results of your experiments, so you can compare its performance later with other models.

That's a lot of dev. Good news : it's called an ML Model Registry, and somebody has already done it for you. Today we are going to use [MLFlow](https://mlflow.org/), but there are some others ([neptune.ai](https://neptune.ai/) or [Amazon SageMaker Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html) for example).

Of course you should use it with an external storage, like a cloud storage, but we are just toying around, it will be easier to store the models locally (plus, this way we don't have to pay for a cloud storage service).

## Organizing your services

Ok now things will get a little bit more _DevOpsy_ and a little lest _data sciencish_. We will need at least two containers :
- one that runs the MLFlow server
- one that runs a PostgreSQL database

Actually, there should be a third container to run your training, but I'll save you this step for this time.
The MLFlow container should be accessible from your notebooks, and the MLFlow server should be able to communicate with the Postgres container.

To do all this locally, you can use Docker Compose. So ... bye bye Python, hello YAML. This what your` docker-compose.yaml` :

```yaml
services:
  db:
    image: "postgres"
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow

  mlflow:
    build: ./mlflow
    ports:
      - "5001:5000"
    expose:
      - 5000

```

```python
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

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
```

Wow ! We now have logged our model in the Mlflow Model Registry, and we tagged it for Production from the training side. It's now time to fetch this model from the registry on the serving side.

```python
import mlflow.pyfunc

model_name = "sk-learn-dummy-model"
model_version = 1

fetched_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

assert (fetched_model.predict(X_test[0:10]) == dummy_classifier.predict(X_test[0:10])).all()
```

Remember, we need to do this in a `serve.py` file. This begs the question : when should we fetch the model. The simplest way (and this is what we are going to do at the step 2 of our journey), is to fetch it at the startup of our webservice. But there are other ways : embedding it in the container at build time, load it from a mounted volume etc, or even use dedicated tools that do all the heavy lifting for you.

```python
# this should be in a file called "serve.py"

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
```

🎉 Tadaa 🎉 ! At the startup of the application, the model is fetched from the MLFlow registry, and you can use it as if it has just been freshly trained !

Now you should package this in a Docker container, but I'll save you (and me) the hassle for this time.

We have not addressed any security issues, but of course, you should be careful with your model registry. Maybe it should be accessed only from your virtual private cloud, or you should use credentials to access MLFlow.

If you are using an external storage service (such as S3 for example), then you should give access to this service to your MFlow container and your application container.

Next time, we will use even more features from MLFlow to:
1. 🏗️ Create a better model
2. 🔍 Compare this model to the one in production
3. 🤖 Automatically tag the better model as the production one
4. 💰 Repeat for infinite money
