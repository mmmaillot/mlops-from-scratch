FROM python:3.10

EXPOSE 8000
WORKDIR /usr/src/app

RUN pip install pipenv
COPY ./main.py .
RUN pipenv install --system --deploy


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]