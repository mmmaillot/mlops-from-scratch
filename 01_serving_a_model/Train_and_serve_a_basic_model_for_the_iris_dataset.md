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

For now, this will be a very simple step. We will be using a scikit-learn dataset, the now famous Iris dataset. It is small, easily downloadable on your laptop, and already cleaned up. Later, we will deal with dirty datasets, too big too load on a single machine.

But for now, we want to focus on the architecture, and the fastest way to shipping a model to production.

```python
from sklearn.datasets import load_iris
iris_dataset =  load_iris()
X,  y = iris_dataset["data"], iris_dataset["target"]
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

See how dumb that model is ? Accuracy is only 39.5%. I'm not even pretending to care. We are going to ship this one, then we will continuously improve this model with the CI until it's (almost) state-of-the-art.

"Regular" code is built by a CI. Nowadays, nobody (I hope), ships code that has been build locally on his machine. Well it should be the same for model training in machine learning. Your model should be trained by a CI (or something equivalent, like Airflow).

For now, the most important thing is shipping this model to prod. Even if it's so bad it could hurt your business. Because even if it's in production, nobody will be actually listening to your endpoint. But it's there, and it's actually the most difficult thing to do.


# Serving a model

There are various ways to serve a model, ie making it available for some external service to call it, and have the model return a prediction or a label.

The most ubiquitous and simple way is just encapsulating it in a webservice. Soooo ... that's what we are going to do now !

## Serving a model (locally)

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
# python base image
FROM python:3.10 

#expose the port 8000 (could be another, but this is the default port for FastAPI)
EXPOSE 8000

# this is the working directory
WORKDIR /usr/src/app

RUN pip install pipenv

COPY ./Pipfile* .
# this is necessary when pipenv run inside a container
RUN pipenv install --deploy

COPY ./main.py .

# the command that will be launched in the container
CMD ["pipenv", "run","uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```


This is not the place (nor I feel qualified) to explain how Docker and the Dockerfile work. But I added some comments line by line.

Now you need to :
1. build the image
2. launch a container with that image

For this, I created for you a little ... Makefile ! Way easier to use than script files. Want to build ? Just type `make build` in your favourite command line.

Want to run the container ? Type `make run`, go to http://localhost:8000/docs, and see how the magic happens.

Want to kill the container ? Type `make kill`.


```make
build:
	docker build . -t basic-model-serve

run:
	docker run --name basic-model-serve --rm -d -p 8000:8000 basic-model-serve

kill:
	docker kill basic-model-serve
```


# Conclusion

In this first example, we have seen how to build a baseline model and encapsulate it in a webservice, and bundle the whole thing in a Docker container. This is a standard practice these days, because of the omnipresence of the cloud.

Of course, there are other ways to use your models, another one would be to make your predictions in batch. Maybe you don't need a real time prediction, maybe you can just label all your dataset during the night, to make it available during the day for your users.

There are a lot of things that could be done in a better way. For example, you should not be training your model at the startup of your webservice. This thing takes time, and your container orchestration service will probably think that your container is not working correctly, and kill it right away. But we will improve every part of this chain in the next chapters.

So, for the next step, we will separate the training of the model, from the serving of the model.
