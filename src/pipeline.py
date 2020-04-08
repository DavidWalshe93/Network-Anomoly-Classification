"""
Author:         David Walshe
Date:           08/04/2020   
"""
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


class CustomColumnTransformer(ColumnTransformer):

    def get_feature_names_from_ohe_step(self):
        return self.named_transformers_['cp'].named_steps["ohe"].get_feature_names()


class PipelineLabelEncoder(TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, X, y=0):
        self.encoder.fit(X)
        return self

    def transform(self, X, y=0):
        return self.encoder.transform(X).reshape(-1, 1)


class PipelineFactory:

    def X_preprocess_pipeline(self, category_variables, numeric_variables):
        return CustomColumnTransformer(
            transformers=[
                ("cp", self._category_step, category_variables),
                ("sp", self._scaler_step, numeric_variables)
            ],
            remainder="drop",
            n_jobs=-1
        )

    def y_preprocess_pipeline(self, variables: tuple = (0,)):
        return ColumnTransformer(
            transformers=[
                ("lep", self._label_encoder_step, variables)
            ],
            remainder="drop",
            n_jobs=-1
        )

    @property
    def _category_step(self) -> Pipeline:
        return Pipeline([
            ("ohe", OneHotEncoder())
        ])

    @property
    def _scaler_step(self) -> Pipeline:
        return Pipeline([
            ("ss", StandardScaler())
        ])

    @property
    def _label_encoder_step(self) -> Pipeline:
        return Pipeline([
            ("le", PipelineLabelEncoder())
        ])
