"""
Author:     David Walshe
Date:       03/04/2020
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Pre-processing
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

from sklearn.model_selection import cross_val_score
from src.data_loader import DataRetriever, LabelManager
from src.timer import Timer

RS_NUMBER = 0


class PipelineLabelEncoder(TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, X, y=0):
        self.encoder.fit(X)
        return self

    def transform(self, X, y=0):
        return self.encoder.transform(X).reshape(-1, 1)


def refactor_names(_names, features):

    for i, feature in enumerate(features):
        for j, name in enumerate(_names):
            if name.find(f"x{i}") > -1:
                name = name.replace(f"x{i}_", f"[{feature}] ")
                name = name.replace("b'", "")
                name = name.replace("'", "")

                _names[j] = name

    return _names


def create_category_pipeline() -> Pipeline:
    return Pipeline([
        ("ohe", OneHotEncoder())
    ])


def create_scaler_pipeline() -> Pipeline:
    return Pipeline([
        ("ss", StandardScaler())
    ])


def create_label_encoder_pipeline() -> Pipeline:
    return Pipeline([
        ("le", PipelineLabelEncoder())
    ])


def plot_model_build_time(stages, times):
    import math
    fig, ax = plt.subplots()
    ax.bar(stages, times)
    plt.xticks(stages, stages)
    max_time = math.ceil(max(times))
    tick_scale = math.ceil(max_time/20)
    max_time += tick_scale
    plt.yticks([i for i in range(0, max_time, tick_scale)], [i if max_time < 60 else f"{int(i/60)}:{i%60}" for idx, i in enumerate(range(0, max_time, tick_scale))])
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    total_time = sum(times)
    if max_time > 60:
        total_time = f"{round(total_time/60)}m {round(total_time%60)}s"
        plt.ylabel("Minutes")
    else:
        plt.ylabel("Seconds")
    plt.xlabel("Stages")

    textstr = f"Total Time: {total_time}"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.show()


if __name__ == '__main__':
    timer = Timer()
    label_manager = LabelManager(config_file="data.json")
    data_retriever = DataRetriever(label_manager=label_manager)
    X, y = data_retriever.X_y_dataset(remove_duplicates=True, full_dataset=True)

    X_discrete = label_manager.X_discreet
    X_continuous = label_manager.X_continuous
    signatures = y["signature"].unique()
    signatures = [str(sig).replace("b'", "") for sig in signatures]
    signatures = [sig.replace(".'", "") for sig in signatures]

    # Define column groups for each pipeline process
    category_variables = X_discrete
    numeric_variables = X_continuous
    timer.time_stage("Data Retrieval")

    X_preprocess_pipeline = ColumnTransformer(
        transformers=[
            ("cp", create_category_pipeline(), category_variables),
            ("sp", create_scaler_pipeline(), numeric_variables)
        ],
        remainder="drop",
        n_jobs=-1
    )

    _X = X_preprocess_pipeline.fit_transform(X)

    names = X_preprocess_pipeline.named_transformers_['cp'].named_steps["ohe"].get_feature_names()
    feature_names = refactor_names(names, category_variables)
    feature_names = np.append(feature_names, X_continuous)

    if type(_X) is not np.ndarray:
        _X = _X.toarray()

    X = pd.DataFrame(data=_X, columns=feature_names)

    timer.time_stage("Preprocess X")

    y_preprocess_pipeline = ColumnTransformer(
        transformers=[
            ("lep", create_label_encoder_pipeline(), [0])
        ],
        remainder="drop",
        n_jobs=-1
    )

    y_np = y["signature"].to_numpy().reshape(-1, 1)
    _y = y_preprocess_pipeline.fit_transform(y_np)

    if type(_y) is not np.ndarray:
        _y = _y.toarray()

    y = pd.DataFrame(data=_y, columns=["signature"])

    timer.time_stage("Preprocess y")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=RS_NUMBER
    )
    y_train_unique = y_train["signature"].unique()
    y_test_unique = y_test["signature"].unique()

    timer.time_stage("Train Test Split")

    # pca = PCA(n_components=None)
    pca = PCA(n_components=10)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    explained_variance = pca.explained_variance_ratio_.reshape(-1, 1)

    perc = explained_variance[0:10].sum()
    timer.time_stage("PCA")

    classifier = SVC(kernel="rbf", random_state=RS_NUMBER)
    timer.time_stage("Building model")

    classifier.fit(X_train, y_train)
    timer.time_stage("Fitting model")

    y_pred = classifier.predict(X_test).reshape(-1, 1)
    timer.time_stage("y Prediction")

    cm = confusion_matrix(y_test, y_pred)
    timer.time_stage("Confusion Matrix")

    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1).reshape(-1, 1)
    mean = accuracies.mean()
    variance = accuracies.std()

    timer.time_stage("Cross Validation")

    timer.time_script()
    stages, times = timer.plot_data
    plot_model_build_time(stages=stages, times=times)

