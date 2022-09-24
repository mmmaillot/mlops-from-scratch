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

```python
from sklearn.datasets import load_iris
```

```python
iris_dataset =  load_iris()
X,  y = iris_dataset["data"], iris_dataset["target"]
```

## Getting a training set and a test set

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
```

## Training a baseline model

We are using a dummy classifier, because the most important thing when thinking MLOps, is not spending a lot of time having the perfect model. The best model is not the one running in your Jupyter Notebook, but the one in production, adding value to your application. So any baseline that is good enough should be shipped to production ASAP.

Since the purpose of these examples is not focusing on ML itself, but MLOps, we will push even further this logic by using the dumbest model possible.

```python
from sklearn.dummy import DummyClassifier

dummy_classifier = DummyClassifier(strategy="constant", constant=0)
```

```python
dummy_classifier.fit(X_train,y_train)
```

```python
accuracy = dummy_classifier.score(X_test, y_test)
f"Accuracy : {100*accuracy:.1f}%"
```

See how dumb that model is ? Accuracy is only 39.5%. I'm not even pretending to care. We are going to ship this one, then CI and CD (more about this in the next examples) will continuously improve this model until it's state-of-the-art.

For now, the most important thing is shipping this model to prod. Even if it's so bad it could hurt your business. Because even if it's in production, nobody will be actually listening to your endpoint. But it's there, and it's actually the most difficult thing to do.
