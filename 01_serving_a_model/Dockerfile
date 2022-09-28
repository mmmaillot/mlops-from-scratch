FROM python:3.10

EXPOSE 8000
WORKDIR /usr/src/app

RUN pip install pipenv

COPY ./Pipfile* .
RUN pipenv install --deploy
COPY ./main.py .


CMD ["pipenv", "run","uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]