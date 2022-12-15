import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import joblib
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
    df_current.drop(["orientation"], axis=1, inplace=True)
    df_baseline.drop(["orientation"], axis=1, inplace=True)
    sd = SmartDrift(
    df_current=df_current,
    df_baseline=df_baseline,
    deployed_model=model, # Optional: put in perspective result with importance on deployed model
    dataset_names={"df_current": "", "df_baseline": ""} # Optional: Names for outputs
    )

    sd.compile(
    full_validation=False, # Optional: to save time, leave the default False value. If True, analyze consistency on modalities between columns.
    date_compile_auc='07/12/2022', # Optional: useful when computing the drift for a time that is not now
    datadrift_file="datadrift_auc_train.csv", # Optional: name of the csv file that contains the performance history of data drift
    )


def train_model(data):
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "price" 
    train_x = train.drop(["price"], axis=1)
    test_x = test.drop(["price"], axis=1)
    train_y = train[["price"]]
    test_y = test[["price"]]

    with mlflow.start_run():
        model = LGBMRegressor()
        model.fit(train_x, train_y)
        predicted_qualities = model.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        df_baseline = pd.read_csv("../data/houses.csv")
        eval_drift(data, df_baseline, model)

        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, "model.joblib")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    data = pd.read_csv("../data/houses.csv")
    
    # Orientations are strings, so we need to convert them to numbers
    data["orientation"] = data["orientation"].map({"Nord": 0, "Est": 1, "Sud": 2, "Ouest": 3})
    train_model(data)