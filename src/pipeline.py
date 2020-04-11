"""
Author:         David Walshe
Date:           08/04/2020   
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

if TYPE_CHECKING:
    import numpy as np
    from pandas import DataFrame


class CustomColumnTransformer(ColumnTransformer):
    """
    Custom ColumnTransformer to allow easy feature extraction
    """

    def get_feature_names_from_ohe_step(self) -> np.ndarray:
        """
        Helper method to access internal step feature names.

        :return: The feature names after the OneHotEncoder step of the Pipeline.
        """
        return self.named_transformers_['cp'].named_steps["ohe"].get_feature_names()


class PipelineLabelEncoder(TransformerMixin):
    """
    Custom LabelEncoder to allow for passing of X and y datasets.

    Default LabelEncoder only accepts one dataset by default and is not
    suitable for Pipeline usage.
    """

    def __init__(self):
        """
        Class Constructor
        """
        self.encoder = LabelEncoder()

    def fit(self, X: DataFrame, y: DataFrame = None):
        """
        Fit the dataset X to the encoder.

        :param X: Passed dataset to be encoded.
        :param y: Dummy variable included to allow for Pipeline usage.
        :return: This instance.
        """
        self.encoder.fit(X)
        return self

    def transform(self, X: DataFrame, y: DataFrame = None) -> np.ndarray:
        """
        Apply the LabelEncoder transformation to the dataset X.

        :param X: The dataset to encode.
        :param y: Dummy variable included to allow for Pipeline usage.
        :return: A numpy ndarray of the applied transformation.
        """
        return self.encoder.transform(X).reshape(-1, 1)

    @property
    def classes_(self):
        return self.encoder.classes_


class PipelineFactory:
    """
    Method Factory Class to help with the creation of various pre-processing Pipelines.
    """

    def X_preprocess_pipeline(self, category_variables: list, numeric_variables: list) -> CustomColumnTransformer:
        """
        Creates a pre-processing pipeline targeted at the X segment for the KDD cup99 dataset.

        :param category_variables: The categorical variable names of the dataset X.
        :param numeric_variables: The numerical variable names of the dataset X.
        :return: A CustomColumnTransformer instance to pre-process the X dataset.
        """
        return CustomColumnTransformer(
            transformers=[
                ("cp", self._category_step, category_variables),
                ("sp", self._scaler_step, numeric_variables)
            ],
            remainder="drop",
            n_jobs=-1
        )

    def y_preprocess_pipeline(self, variables: tuple = (0,)):
        """
        Creates a pre-processing pipeline targeted at the y segment for the KDD cup99 dataset.

        :param variables: Optional argument to pass in column indexes to use in the pipeline. Default= (0, )
        :return: A CustomColumnTransformer instance to pre-process the X dataset.
        """
        return ColumnTransformer(
            transformers=[
                ("lep", self._label_encoder_step, variables)
            ],
            remainder="drop",
            n_jobs=-1
        )

    @property
    def _category_step(self) -> Pipeline:
        """
        Property to get the category step for use in a Pipeline.

        :return: Pipeline with an OneHotEncoder internal step.
        """
        return Pipeline([
            ("ohe", OneHotEncoder())
        ])

    @property
    def _scaler_step(self) -> Pipeline:
        """
        Property to get the scaler step for use in a Pipeline.

        :return: Pipeline with an StandardScaler internal step.
        """
        return Pipeline([
            ("ss", StandardScaler())
        ])

    @property
    def _label_encoder_step(self) -> Pipeline:
        """
        Property to get the encoder step for use in a Pipeline.

        :return: Pipeline with a LabelEncoder internal step.
        """
        return Pipeline([
            ("le", PipelineLabelEncoder())
        ])
