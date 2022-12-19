# MLOPS

## AUTHORS

Alexandre Lemonnier & Victor Simonin

---

## Subject

In this project you will have to put into production a machine learning model of your choice.

The deployment of the model must be done automatically via a simple script.

The model will need to be able to handle some load. You can either do batch processing / an HTTP API or streaming.

The system must be able to:

either to retrain on their own regularly thanks to newly labeled data
either to raise alerts if there are risks that the model no longer works (distribution shift)

Bonus: the model will be packaged in a docker container or will be deployed via Kubernetes / kubeflow

---

## Usage

### Docker

You can build the docker images with the following command:

```bash
docker-compose up -d --build
```

Once the docker images are built and started, you can access to the following services:
- the streamlit front-end at the following address [http://localhost:8501](http://localhost:8501).
- the web interface of the mlflow server at the following address [http://localhost:5000](http://localhost:5000).
- the uvicorn api at the following address [http://localhost:8000](http://localhost:8000).

To shutdown the docker containers, run the following command:

```bash
docker-compose down -v
```

### Manual

The training of the model is in the `src/train_model.py` file. It is possible to train the model with the following command:

```bash
cd src
python train_model.py
```

A mlflow workflow is available to track the training. After a few training, to start the mlflow server, run the following command:

```bash
mlflow ui
```

To start the api, run the following command:

```bash
cd src
uvicorn api:app --reload
```

The streamlit front-end is available with the following command:

```bash
cd src
streamlit run model_app.py
```
