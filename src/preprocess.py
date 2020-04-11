"""
Author:         David Walshe
Date:           09/04/2020   
"""

import pandas as pd
import numpy as np

from src.data import LabelManager
from src.pipeline import PipelineFactory
from src.utils import refactor_names, refactor_byte_name


class Preprocess:

    def __init__(self):
        self._signature_keys = None

    def X_pre_process(self, X, pipeline_factory: PipelineFactory, **kwargs):
        X_preprocess_pipeline = pipeline_factory.X_preprocess_pipeline(**kwargs)

        _X = X_preprocess_pipeline.fit_transform(X)

        names = X_preprocess_pipeline.get_feature_names_from_ohe_step()
        feature_names = refactor_names(names, kwargs["category_variables"])
        feature_names = np.append(feature_names, kwargs["numeric_variables"])

        _X = self._convert_to_array(_X)

        X = pd.DataFrame(data=_X, columns=feature_names)

        return X

    def y_pre_process(self, y, pipeline_factory: PipelineFactory):
        y_preprocess_pipeline = pipeline_factory.y_preprocess_pipeline()
        y = y_preprocess_pipeline.fit_transform(y)
        y = self._convert_to_array(y)
        y = pd.DataFrame(data=y, columns=["signature"])
        self._signature_keys = y_preprocess_pipeline.named_transformers_['lep'].named_steps["le"].classes_

        return y

    def X_y_pre_process(self, X, y, label_manager: LabelManager):
        pipeline_factory = PipelineFactory()

        X = self.X_pre_process(X, pipeline_factory,
                               category_variables=label_manager.X_discrete,
                               numeric_variables=label_manager.X_continuous)
        y = self.y_pre_process(y, pipeline_factory)

        return X, y

    @staticmethod
    def _convert_to_array(dataset: pd.Series):
        if type(dataset) is not np.ndarray:
            dataset = dataset.toarray()

        return dataset

    @property
    def y_classes(self):
        return {key: refactor_byte_name(value) for key, value in enumerate(self._signature_keys)}