# MLOPS

In this project you will have to put into production a machine learning model of your choice.

The deployment of the model must be done automatically via a simple script.

The model will need to be able to handle some load. You can either do batch processing / an HTTP API or streaming.

The system must be able to:

either to retrain on their own regularly thanks to newly labeled data
either to raise alerts if there are risks that the model no longer works (distribution shift)

Bonus: the model will be packaged in a docker container or will be deployed via Kubernetes / kubeflow

---

## Usage

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