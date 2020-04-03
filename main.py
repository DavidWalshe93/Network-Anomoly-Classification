"""
Author:     David Walshe
Date:       03/04/2020
"""

import numpy as np
import pandas as pd

# Pre-processing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.data_retriever import get_dataset_as_X_y


RS_NUMBER = 0

if __name__ == '__main__':
    X, y = get_dataset_as_X_y()

    proto_type = X["land"].unique()

    # ct = ColumnTransformer(
    #     [("one_hot_encoder", OneHotEncoder(), [0])],
    #     remainder="passthrough"
    # )
    #
    # X_columns = get_column_names()[:-1]
    #
    # ids = []
    # indexes = []
    #
    # for idx, name in enumerate(X_columns):
    #     unique_items = X[name].unique()
    #
    #     if len(unique_items) > 2:
    #         if type(unique_items[0]) is bytes:
    #             ids.append(name)
    #             indexes.append(idx)

