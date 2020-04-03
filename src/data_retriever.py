"""
Author:         David Walshe
Date:           03/04/2020   
"""

# Imports
from sklearn.datasets import fetch_kddcup99

import numpy as np
import pandas as pd
from json import load


def get_data_info() -> dict:
    with open("data.json") as fh:
        data_info = load(fp=fh)

    return data_info


def _get_column_names(key: str, data_info: dict) -> list:
    return [item.get("name") for item in data_info.get(key)]


def get_column_names_X(data_info: dict) -> list:
    return _get_column_names(key="X", data_info=data_info)


def get_column_names_y(data_info: dict) -> list:
    return _get_column_names(key="y", data_info=data_info)


def _get_column_names_X_y() -> tuple:
    """
    Helper function for acquiring the dataset column names from data.json.

    :return: The X columns names and the y column name as a tuple.
    """
    data_info = get_data_info()

    columns_X = get_column_names_X(data_info=data_info)
    columns_y = get_column_names_y(data_info=data_info)

    return columns_X, columns_y


def get_dataset_as_X_y() -> np.array:
    """
    Helper function to create the dataset, including the dependant "target" variable.

    :return: The dataset as an np.array, including the dependant variable.
    """
    data, target = fetch_kddcup99(return_X_y=True)

    target = np.array(target).reshape(-1, 1)

    X_columns, y_column = _get_column_names_X_y()

    X = pd.DataFrame(data=data, columns=X_columns)
    y = pd.DataFrame(data=target, columns=y_column)

    return X, y
