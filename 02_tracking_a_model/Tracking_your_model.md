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