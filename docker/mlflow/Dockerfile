FROM python:3.9
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
COPY data /app/data
COPY src /app/src
WORKDIR /app/src
EXPOSE 5000
ENTRYPOINT [ "mlflow", "ui", "--host", "0.0.0.0" ]
