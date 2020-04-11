"""
Author:     David Walshe
Date:       03/04/2020
"""
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime

# Pre-processing
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from src.data import DataRetriever, LabelManager
from src.plot import plot_model_build_time
from src.preprocess import Preprocess
from src.timer import Timer
from itertools import zip_longest
import random

mpl.rcParams['figure.dpi'] = 300

RS_NUMBER = 0

import random


def get_colors():
    import json
    with open("colors.json") as fh:
        colors = json.load(fh)

    return colors.values()


def plot_2d_space(X, y, label='Classes'):
    r = lambda: random.randint(0, 255)
    colors = get_colors()
    # markers = ['o', 's']
    for l, c in zip_longest(np.unique(y), colors):
        plt.scatter(
            X[y == l, 0],
            X[y == l, 1],
            c=c, label=l, marker="o"
        )
    plt.title(label)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_value_counts(y, title="Count (target)"):
    if type(y) is not pd.DataFrame:
        y = pd.DataFrame(y, columns=["signature"])

    target_count = y["signature"].value_counts()
    target_count.plot.bar(title=title)
    plt.show()


def get_y_distribution(y):
    distribution = pd.DataFrame(columns=["class", "count", "percentage"])
    counter = Counter(y["signature"])
    for index, (k, v) in enumerate(counter.items()):
        per = v / len(y) * 100
        distribution.loc[index] = {"class": preprocess.y_classes.get(k, k), "count": v, "percentage": per}

    return distribution.sort_values("percentage", ascending=False)


if __name__ == '__main__':
    # Setup
    # =====
    timer = Timer()
    label_manager = LabelManager(config_file="data.json")
    data_retriever = DataRetriever(label_manager=label_manager)

    # Get Raw Data.
    # =============
    X, y = data_retriever.X_y_dataset(remove_duplicates=True, full_dataset=False)
    timer.time_stage("Data Retrieval")

    # Preprocess raw data
    # ===================
    preprocess = Preprocess()
    X, y = preprocess.X_y_pre_process(X, y, label_manager)
    timer.time_stage("Preprocess step")

    # Test/Train Split
    # ================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=RS_NUMBER
    )

    # Data Analysis
    # =============
    plot_value_counts(y_train)
    y_dis_init = get_y_distribution(y_train)

    # # PCA 2-D
    # pca = PCA(n_components=10)
    # X_train = pca.fit_transform(X_train)
    #
    # over = RandomOverSampler(sampling_strategy="minority")
    #
    # X_train, y_train = over.fit_sample(X_train, y_train)
    #
    # under = RandomUnderSampler(sampling_strategy="majority")
    #
    # X_train, y_train = under.fit_sample(X_train, y_train)
    #
    # model = DecisionTreeClassifier()
    #
    # model = model.fit(X_train, y_train)
    #
    # pipeline = Pipeline(steps=[("o", over), ("u", under), ("m", model)])
    #
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    #
    # scores = cross_val_score(pipeline, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
    #
    # print(f"Mean ROC AUC: {np.mean(scores)}")

    #
    # if type(y) is np.ndarray:
    #     y_train = y_train.reshape(-1, 1).ravel()
    # else:
    #     y_train = y_train.to_numpy().reshape(-1, 1).ravel()
    #
    # if type(X_train) is not np.ndarray:
    #     X_train = X_train.to_numpy()
    #
    # plot_2d_space(X_train, y_train, 'Imbalanced dataset (2 PCA components)')
    #
    # # Under-Sampling
    # # =============
    # from imblearn.under_sampling import RandomUnderSampler
    #
    # rus = RandomUnderSampler()
    # X_rus, y_rus = rus.fit_sample(X_train, y_train)
    #
    # plot_2d_space(X_rus, y_rus, 'Random under-sampling')
    # plot_value_counts(y_rus)
    #
    # # Over-Sampling
    # # =============
    # from imblearn.over_sampling import RandomOverSampler
    #
    # ros = RandomOverSampler()
    # X_ros, y_ros = ros.fit_sample(X_train, y_train)
    #
    # plot_2d_space(X_ros, y_ros, 'Random Over-Sampling')
    # plot_value_counts(y_ros)
    #
    # # TomekLinks
    # # ==========
    # from imblearn.under_sampling import TomekLinks, RepeatedEditedNearestNeighbours
    # from collections import Counter
    #
    # # for arg in ("majority", "not majority", "not minority", "all", "auto"):
    # arg = "all"
    # X_u = X_train
    # y_u = y_train
    # print(f"Original y size: {Counter(y_u)}")
    # for i in range(10):
    #     renn = RepeatedEditedNearestNeighbours(n_jobs=-1)
    #     X_u, y_u = renn.fit_resample(X_u, y_u)
    #     # tl = TomekLinks(sampling_strategy=arg, n_jobs=-1)
    #     # X_u, y_u, = tl.fit_sample(X_u, y_u)
    #     if i % 10 == 0:
    #         plot_value_counts(y_u, title=f"Count ({arg})")
    #     # plot_2d_space(X_tl, y_tl, f'Tomek links under-sampling ({arg})')
    #     # plot_value_counts(y_tl, title=f"Count ({arg})")
    #     print(f"Run {i} y Counter: {Counter(y_u)}")
    #     print(f"Run {i} y size: {y_u.size}")

    # from sklearn.datasets import make_classification
    #
    # _X, _y = make_classification(
    #     n_classes=2, class_sep=1.5, weights=[0.9, 0.1],
    #     n_informative=3, n_redundant=1, flip_y=0,
    #     n_features=20, n_clusters_per_class=1,
    #     n_samples=100, random_state=10
    # )
    #
    # pca = PCA(n_components=2)
    # _X = pca.fit_transform(_X)
    #
    # plot_2d_space(_X, _y, 'Imbalanced dataset (2 PCA components)')
    #
    # from imblearn.under_sampling import ClusterCentroids
    #
    # cc = ClusterCentroids(sampling_strategy={key: 1 for key in (int(i) for i in np.unique(y))})
    # X_cc, y_cc = cc.fit_sample(X_train, y_train)
    #
    # plot_2d_space(X_cc, y_cc, 'Cluster Centroids under-sampling')

    # from imblearn.under_sampling import TomekLinks
    #
    # for arg in ("majority", "not majority", "not minority", "all", "auto"):
    #     print("*" * 10)
    #     print(arg)
    #     tl = TomekLinks(sampling_strategy=arg)
    #     X_tl, y_tl = tl.fit_sample(_X, _y)
    #
    #     print('Removed indexes:', tl.sample_indices_)
    #
    #     print(f"t1 size: {tl.sample_indices_.size}")
    #     print(f"_y size: {_y.size}")
    #     print(f"diff size: {_y.size - tl.sample_indices_.size}")
    #
    #     plot_2d_space(X_tl, y_tl, f'Tomek links under-sampling ({arg})')

    # y_train_unique = y_train["signature"].unique()
    # y_test_unique = y_test["signature"].unique()

    timer.time_stage("Train Test Split")

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
    # stages, times = timer.plot_data
    # plot_model_build_time(stages=stages, times=times)
