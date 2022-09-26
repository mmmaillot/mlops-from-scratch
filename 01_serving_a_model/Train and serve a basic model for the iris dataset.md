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

# Creating a model

## Getting a dataset

For now, this will be a very simple step. We will be using a scikit-learn dataset, the now famous Iris dataset. It is small, easily downloadable on your laptop, already cleaned up. Later, we will deal with dirty datasets, too big too load on a single machine.

But for now, we want to focus on the architecture, and the fastest way to shipping a model to production.

```python
from sklearn.datasets import load_iris
iris_dataset =  load_iris()
X,  y = iris_dataset["data"], iris_dataset["target"]
```

```python
iris_dataset
```

## Getting a training set and a test set

The purpose of this course is not to focus on ML best practices, but you have to understand at least the basics of it. Dividing your dataset in train, test and validation will have a great influence on the final architecture of your MLOps processes.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
```

## Training a baseline model

We are using a dummy classifier, because the most important thing when thinking MLOps, is not spending a lot of time having the perfect model. The best model is not the one running in your Jupyter Notebook, but the one in production, adding value to your application. So any baseline that is good enough should be shipped to production ASAP.

Since the purpose of these examples is not focusing on ML itself, but MLOps, we will push even further this logic by using the dumbest model possible.

```python
from sklearn.dummy import DummyClassifier

dummy_classifier = DummyClassifier(strategy="constant", constant=y[0])
dummy_classifier.fit(X_train,y_train)

accuracy = dummy_classifier.score(X_test, y_test)
f"Accuracy : {100*accuracy:.1f}%"
```

See how dumb that model is ? Accuracy is only 39.5%. I'm not even pretending to care. We are going to ship this one, then CI and CD (more about this in the next examples) will continuously improve this model until it's state-of-the-art.

For now, the most important thing is shipping this model to prod. Even if it's so bad it could hurt your business. Because even if it's in production, nobody will be actually listening to your endpoint. But it's there, and it's actually the most difficult thing to do.


## Serving a model

There are various ways to serve a model, ie making it available for some external service to call it, and have the model return a prediction or a label.

The most ubiquitous and simple way is just encapsulating it in a webservice. Soooo ... that's what we are going to do now !

```python
from fastapi import FastAPI
from pydantic import BaseModel


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
```

You could just run this code snippet, go to http://127.0.0.1:8000/docs and toy around with the API (and see how cool FastAPI is), but if you want to have a quick feedback loop, it is best to have small unit tests.

Here is one for example :

```python
from fastapi.testclient import TestClient


def test_is_response_has_correct_form():
    client = TestClient(app)
    response = client.post("/label",
                           json={"sepal_length": 0,
                                 "sepal_width": 0,
                                 "petal_length": 0,
                                 "petal_width": 0}).json()
    assert "label" in response
    assert response["label"] in [0, 1, 2]
```

We'll see later how we can leverage the different kinds of tests and how we can leverage them to improve our MLOps feedback loop.


## Serving a model (for real)

If you're a data scientist reading this, serving your model might be a huge step outside of your regular attributions.

But this is not how grown-ups serve their models. Time to take an other extra step. Yes, we are going to bundle it in a Docker image.


```Dockerfile
FROM python:3.10

EXPOSE 8000
WORKDIR /usr/src/app

RUN pip install pipenv
COPY ./main.py .
RUN pipenv install --system --deploy


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```


This is not the place (nor I feel qualified) to explain how Docker and the Dockerfile work. But to sum up quickly line by line :
* We use the ubiquitous python base image
* We expose the port 8000 (could be another, but this is the default port for FastAPI)

