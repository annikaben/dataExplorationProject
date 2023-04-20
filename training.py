import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
from pathlib import Path

def importDf() -> pd.DataFrame:
    """
    imports the prepared dataframe from the file

    :return: returns the imported data as pandas dataframe
    """
    csv = Path("./data/datasetClean.csv")
    if csv.is_file():
        dfImported = pd.read_csv("./data/datasetClean.csv")
    else:
        dfImported = pd.read_csv("datasetClean.csv")
    return dfImported

importDf()


def splitDf(df_tosplit: pd.DataFrame) -> list[pd.DataFrame]:
    """
    splits data into the three different dataframes testing (10%), validation (20%) and training (70%)
    and then splits the training and validation data into the determinating factors and the target.

    :param df_tosplit: complete dataset
    :return: returns all splitted parts of the previous dataset
    """
    df_train, df_test = train_test_split(df_tosplit, test_size=0.3, random_state=10)
    df_validate, df_test = train_test_split(df_test, test_size=1 / 3, random_state=10)

    train_x = df_train.drop(["Target"], axis=1)
    val_x = df_validate.drop(["Target"], axis=1)
    train_y = df_train[["Target"]]
    val_y = df_validate[["Target"]]
    return [train_x, val_x, train_y, val_y, df_test]


def evaluation_metrics(validation: pd.DataFrame, prediction: np.ndarray) -> list[float]:
    """
    calculates metrics for the usability of the model and returns them.

    :param validation: validation dataset
    :param prediction: prediction of the model for the data
    :return: returns different metrics for the evaluation
    """
    rmse = np.sqrt(mean_squared_error(validation, prediction))
    mae = mean_absolute_error(validation, prediction)
    r2 = r2_score(validation, prediction)
    return [rmse, mae, r2]


def beginTraining(df):
    """
    retrieves all neccessary data for training and starts the loop for training multiple models with different regulizers

    :param df: full dataset to be split for the training
    """
    train_x, val_x, train_y, val_y, unused_test = splitDf(df)

    for n in range (1, 9, 1):
        trainElasticNet(train_x, val_x, train_y, val_y, n)


def trainElasticNet(train_x, val_x, train_y, val_y, i):
    """
    carries out the training of the model with elastic net regression

    :param train_x: training dataset without the target
    :param val_x: dataset without the target to validate the models prediction
    :param_train_y: training dataset consisting of the values of the target 
    :param val_y: dataset consisting of the values of the target to allow evaluation of the models accuracy
    :param i: changing value of the regularization parameter C
    """
    with mlflow.start_run():
        lr_model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=i/10, C=i, max_iter=1000)
        lr_model.fit(train_x, train_y)
        prediction = lr_model.predict(val_x)

        rmse, mae, r2 = evaluation_metrics(val_y, prediction)
        print("Logistic Regression with Elasticnet (l1_ratio={:f}, C={:f}):".format(i/10, i))
        print(" rmse: %s" % rmse)
        print(" mae: %s" % mae)
        print(" r2: %s" % r2)

        mlflow.log_params({
            'l1_ratio': i/10,
            'C': i
        })
        mlflow.log_metrics({
            'rmse': rmse,
            'r2': r2,
            'mae': mae
        })
        mlflow.sklearn.log_model(lr_model, "model")

beginTraining(importDf())