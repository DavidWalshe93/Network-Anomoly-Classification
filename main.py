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
from src.data import DataRetriever, LabelManager
from src.plot import plot_model_build_time
from src.timer import Timer


RS_NUMBER = 0





if __name__ == '__main__':
    timer = Timer()

    label_manager = LabelManager(config_file="data.json")
    data_retriever = DataRetriever(label_manager=label_manager)

    # Get Raw Data.
    # =============
    X, y = data_retriever.X_y_dataset(remove_duplicates=True, full_dataset=False)
    timer.time_stage("Data Retrieval")

    # Preprocess raw data.
    # ====================
    X, y = X_y_pre_process(X, y, label_manager)
    timer.time_stage("Preprocess step")

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=.20, random_state=RS_NUMBER
    # )
    # y_train_unique = y_train["signature"].unique()
    # y_test_unique = y_test["signature"].unique()
    #
    # timer.time_stage("Train Test Split")
    #
    # # pca = PCA(n_components=None)
    # pca = PCA(n_components=10)
    #
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)
    #
    # explained_variance = pca.explained_variance_ratio_.reshape(-1, 1)
    #
    # perc = explained_variance[0:10].sum()
    # timer.time_stage("PCA")
    #
    # classifier = SVC(kernel="rbf", random_state=RS_NUMBER)
    # timer.time_stage("Building model")
    #
    # classifier.fit(X_train, y_train)
    # timer.time_stage("Fitting model")
    #
    # y_pred = classifier.predict(X_test).reshape(-1, 1)
    # timer.time_stage("y Prediction")
    #
    # cm = confusion_matrix(y_test, y_pred)
    # timer.time_stage("Confusion Matrix")
    #
    # accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1).reshape(-1, 1)
    # mean = accuracies.mean()
    # variance = accuracies.std()
    #
    # timer.time_stage("Cross Validation")

    timer.time_script()
    stages, times = timer.plot_data
    plot_model_build_time(stages=stages, times=times)
