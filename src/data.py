"""
Author:         David Walshe
Date:           03/04/2020   
"""

# Imports
import logging
from json import load

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99

from src.logger_config import setup_logger

logger = setup_logger(logging.getLogger(__name__))


class LabelManager:

    def __init__(self, config_file="data.json"):
        self._data_info = self._read_data_info(config_file=config_file)
        self._X_column_names = None
        self._y_column_name = None

    @staticmethod
    def _read_data_info(config_file) -> dict:
        with open(config_file) as fh:
            data_info = load(fp=fh)

        return data_info

    @property
    def info(self):
        return self._data_info

    def _get_column_names(self, key: str) -> list:
        return [item.get("name") for item in self.info.get(key)]

    @property
    def X_column_names(self):

        # Lazy Init
        if self._X_column_names is None:
            self._X_column_names = self._get_column_names(key="X")

        return self._X_column_names

    @property
    def y_column_name(self):

        # Lazy Init
        if self._y_column_name is None:
            self._y_column_name = self._get_column_names(key="y")

        return self._y_column_name

    @property
    def X_y_column_names(self) -> tuple:
        """
        Helper function for acquiring the dataset column names from data.json.

        :return: The X columns names and the y column name as a tuple.
        """
        return self.X_column_names, self.y_column_name

    def get_variable_on_dtype(self, key: str, dtype: str) -> list:
        return [item.get("name") for item in self.info.get(key) if item.get("dtype") == dtype]

    @property
    def X_discrete(self):
        return self.get_variable_on_dtype(key="X", dtype="discrete")

    @property
    def X_continuous(self):
        return self.get_variable_on_dtype(key="X", dtype="continuous")


class DataRetriever:

    def __init__(self, label_manager: LabelManager):
        self.label_manager = label_manager
        self._X = None
        self._y = None

    def _remove_duplicate_rows(self):

        dataset = pd.concat([self.X, self.y], axis=1, join="outer")
        logger.info(f"Step  - Original dataset record count: {dataset.shape[0]}")

        dataset_reduced = dataset.drop_duplicates(inplace=False)
        logger.info(f"Step  - Dataset record count with duplicates removed: {dataset_reduced.shape[0]}")
        logger.info(
            f"Step  - Dataset records reduced by {round(100 - ((dataset_reduced.shape[0] / dataset.shape[0]) * 100), 2)}%")

        self._X = pd.DataFrame(data=dataset_reduced.iloc[:, :-1].values, columns=self.label_manager.X_column_names)

        self._y = pd.DataFrame(data=dataset_reduced.iloc[:, -1].values.reshape(-1, 1),
                               columns=self.label_manager.y_column_name)

    def X_y_dataset(self, remove_duplicates=False, full_dataset=True) -> np.array:
        """
        Helper function to create the dataset, including the dependant "target" variable.

        :return: The dataset as an np.array, including the dependant variable.
        """
        # Lazy init
        if self._X is None or self._y is None:

            logger.info(f"Step - Only 10% of Dataset: {(not full_dataset)}")
            data, target = fetch_kddcup99(return_X_y=True, percent10=(not full_dataset))

            target = np.array(target).reshape(-1, 1)

            self._X = pd.DataFrame(data=data, columns=self.label_manager.X_column_names)
            self._y = pd.DataFrame(data=target, columns=self.label_manager.y_column_name)

            if remove_duplicates:
                self._remove_duplicate_rows()

        return self._X, self._y

    @property
    def X(self):

        # Lazy init
        if self._X is None:
            self.X_y_dataset()

        return self._X

    @property
    def y(self):

        # Lazy init
        if self._y is None:
            self.X_y_dataset()

        return self._y
