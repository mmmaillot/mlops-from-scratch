FROM python:3.10

EXPOSE 5000

RUN pip install mlflow psycopg2

CMD ["mlflow", "server", "--host", "0.0.0.0", "--backend-store-uri", "postgresql://mlflow:mlflow@db:5432", "--default-artifact-root", "./artifacts"]