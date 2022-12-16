FROM python:3.9
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
COPY data /app/data
COPY src /app/src
WORKDIR /app/src
EXPOSE 8000
ENTRYPOINT [ "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]