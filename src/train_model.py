import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import joblib
import os
import mlflow
import mlflow.sklearn
from eurybia import SmartDrift

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def eval_drift(df_current, df_baseline, model):
    sd = SmartDrift(
        df_current=df_current,
        df_baseline=df_baseline,
        deployed_model=model,
        dataset_names={"df_current": "", "df_baseline": ""}
    )

    sd.compile(
        full_validation=False,
        datadrift_file="datadrift_auc_train.csv",
    )
    datadrift_df = pd.read_csv("datadrift_auc_train.csv")
    return datadrift_df['auc'].iloc[-1]


def train_model(data):
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "price"
    train_x = train.drop(["price"], axis=1)
    test_x = test.drop(["price"], axis=1)
    train_y = train[["price"]]
    test_y = test[["price"]]

    with mlflow.start_run():
        if os.path.exists("model.joblib"):
            model = joblib.load("model.joblib")
        else:
            model = LGBMRegressor()
        model.fit(train_x, train_y)
        predicted_qualities = model.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        df = None
        if os.path.exists("../data/new_houses.csv"):
            df = pd.read_csv("../data/new_houses.csv")
            df_baseline = df.copy()
            df_baseline.drop(["price"], axis=1, inplace=True)
        else:
            df_baseline = train_x
        drift = eval_drift(train_x, df_baseline, model)
        
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        print("  AUC: %s" % drift)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("auc", drift)

        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, "model.joblib")
        if df is None:
            df = data
        else:
            df = pd.concat([df, data], axis=0)
        df.to_csv("../data/new_houses.csv", index=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    data = pd.read_csv("../data/houses.csv")

    # Orientations are strings, so we need to convert them to numbers
    data["orientation"] = data["orientation"].map(
        {"Nord": 0, "Est": 1, "Sud": 2, "Ouest": 3})
    train_model(data)
