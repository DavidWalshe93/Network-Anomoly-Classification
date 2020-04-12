"""
Author:     David Walshe
Date:       03/04/2020
"""
import logging
import random
from collections import Counter
from itertools import zip_longest

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Pre-processing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from src.data import DataRetriever, LabelManager
from src.evaluate import ModelEvaluator
from src.logger_config import setup_logger
from src.pipeline import SamplingPipelineFactory
from src.plot import plot_model_build_time
from src.preprocess import Preprocess
from src.timer import Timer

mpl.rcParams['figure.dpi'] = 150

RS_NUMBER = 0

logger = setup_logger(logging.getLogger(__name__))


def get_colors():
    import json
    with open("colors.json") as fh:
        colors = json.load(fh)

    return list(colors.values())


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
    distribution = pd.DataFrame(columns=["label", "class", "count", "percentage"])
    counter = Counter(y["signature"])
    for index, (k, v) in enumerate(counter.items()):
        per = v / len(y) * 100
        distribution.loc[index] = {"label": k, "class": preprocess.y_classes.get(k, k), "count": v, "percentage": per}

    return distribution.sort_values("percentage", ascending=False)


# def eval_model(model, X, y) -> np.ndarray:
#     cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
#     if type(y) is pd.DataFrame:
#         y = y.to_numpy().ravel()
#     scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
#
#     return scores
#
#
#
#
#
# def get_class_for_label(label):
#     return preprocess.y_classes.get(label)
#
#
# def get_all_classes_from_labels():
#     return [preprocess.y_classes.get(idx) for idx, label in enumerate(preprocess.y_classes)]
#
#
# def run_model_evaluation(X_train, y_train, X_test, y_test):
#     logger.info("Stage - Model Analysis BEGIN")
#     results = []
#     models = create_model_collection(n_estimators=100)
#     cms = []
#     names = []
#     for idx, (name, model) in enumerate(models.items()):
#         names.append(name)
#         logger.info(f"Step  - Fitting {name} model BEGIN")
#         model.fit(X_train, y_train.to_numpy().ravel())
#         models[name] = model
#         logger.info(f"Step  - Fitting {name} model END {timer.time_stage(f'Fitting Model {name}')}")
#
#         logger.info(f"Step  - Evaluating {name} model BEGIN")
#         scores = eval_model(model, X_test, y_test.to_numpy().ravel())
#         results.append(scores)
#         logger.info(f"Step  - Evaluating {name} model {timer.time_stage(f'{name} Evaluation')} END")
#
#         logger.info(f"Step  - Prediction BEGIN")
#         y_pred = model.predict(X_test).reshape(-1, 1)
#         logger.info(f"Step  - Prediction END {timer.time_stage(f'y Prediction {name}')}")
#         timer.time_stage(f"y Prediction {name}")
#
#         logger.info(f"Step  - Confusion Matrix BEGIN")
#         cms.append(confusion_matrix(y_test, y_pred))
#         logger.info(f"Step  - Confusion Matrix END {timer.time_stage(f'Confusion Matrix {name}')}")
#
#     [logger.info(f"Step  - Results: {name} - mean={mean(scores):0.3f}, std={std(scores):0.3f}") for name, scores in zip(names, results)]
#
#     # plot the results
#     plt.boxplot(results, labels=list(models.keys()), showmeans=True)
#     plt.show()
#
#     for name, cm in zip(names, cms):
#         div = np.sum(cm, axis=1)
#         div_inv = div.reshape(-1, 1)
#         cm_div = (cm / div_inv[None, :])
#         cm_div = np.nan_to_num(cm_div, posinf=0.0, neginf=0.0)
#         plt.subplots(figsize=(20, 10))
#         ax: plt.Axes = sns.heatmap(cm_div[0], cmap="YlOrRd", linewidths=0.5, annot=True, xticklabels=get_all_classes_from_labels(), yticklabels=get_all_classes_from_labels(), fmt=".2f", cbar=False)
#         ax.set_title(f"Confusion matrix ({name})")
#         ax.set_xlabel("Predictions")
#         ax.set_ylabel("Real Values")
#         plt.show()
#
#     logger.info("Stage - Model Analysis END")


if __name__ == '__main__':
    logger.info("Start")

    # Setup
    # =====
    timer = Timer()
    logger.info("Stage - Data Retrieval BEGIN")
    label_manager = LabelManager(config_file="data.json")
    data_retriever = DataRetriever(label_manager=label_manager)

    # Get Raw Data.
    # =============
    logger.info("Stage -  BEGIN")
    X, y = data_retriever.X_y_dataset(remove_duplicates=True, full_dataset=False)
    logger.info(f"Stage - Data Retrieval END {timer.time_stage('Data Retrieval')}")

    # Preprocess raw data
    # ===================
    logger.info("Stage - Preprocess BEGIN")
    preprocess = Preprocess()
    X, y = preprocess.X_y_pre_process(X, y, label_manager)
    logger.info(f"Stage - Preprocess END {timer.time_stage('Preprocessing')}")

    # Principle Component Analysis
    # ============================
    logger.info("Stage - PCA BEGIN")
    X_backup = X

    # PCA with 3 Components
    # =====================
    if True is False:
        logger.info("Stage - PCA 3 Component BEGIN")
        X = X_backup
        pca = PCA(n_components=3)
        X = pca.fit_transform(X)
        explained_variance_3d = pca.explained_variance_ratio_.reshape(-1, 1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y["signature"], marker='o', cmap=plt.cm.get_cmap('tab20', 23))

        ax.set_title("PCA Analysis (3 Components)")
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')

        plt.show()

        logger.info(f"Step  - PCA 3 Component END {timer.time_stage('PCA 3C')}")

    # PCA with 2 Components
    # =====================
    logger.info("Stage - PCA 2 Component BEGIN")
    X = X_backup
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    explained_variance_2d = pca.explained_variance_ratio_.reshape(-1, 1)

    plt.scatter(X[:, 0], X[:, 1],
                c=y["signature"], edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('tab20', 23))
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.show()
    logger.info(f"Step  - PCA 2 Component END {timer.time_stage('PCA 2C')}")
    logger.info("Stage - PCA END")

    # Test/Train Split
    # ================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=RS_NUMBER
    )
    timer.time_stage("Train Test Split")

    plot_value_counts(y_test, title="X_test Distribution")

    # # Data Analysis - Base Line
    # # =========================
    # plot_value_counts(y)
    # y_dis_init = get_y_distribution(y)
    #
    # model = DummyClassifier(strategy="most_frequent")
    #
    # scores = eval_model(model, X, y)
    # print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    # timer.time_stage("Base Line Analysis")

    # Sampling
    # ========
    logger.info("Stage - Sampling BEGIN")
    sampling_pipeline = SamplingPipelineFactory(y_train, max_sample_limit=5000).sampling_pipeline()

    X_train, y_train = sampling_pipeline.fit_resample(X_train, y_train)

    y_sam_dist = get_y_distribution(y_train)

    plt.scatter(X_train[:, 0], X_train[:, 1],
                c=y_train["signature"], edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('tab20', 23))
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.show()
    logger.info(f"Stage - Sampling END {timer.time_stage('Sampling')}")

    # Testing Model Performance
    # =========================
    model_evaluator = ModelEvaluator(preprocess.y_classes)

    model_evaluator.run_model_evaluation(X_train, y_train, X_test, y_test)

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

    logger.info("Complete")
