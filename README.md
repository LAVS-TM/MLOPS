# MLOPS [![Profile][title-img]][profile]

[title-img]:https://img.shields.io/badge/-LAVS-blue
[profile]:https://github.com/LAVS-TM

This project describe a complete machine learning project in production, with the training and the deployement phase.

## AUTHORS

Alexandre Lemonnier & Victor Simonin

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
