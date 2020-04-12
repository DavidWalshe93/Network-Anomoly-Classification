"""
Author:         David Walshe
Date:           12/04/2020   
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import mean, std
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC

from src.logger_config import setup_logger
from src.timer import Timer

logger = setup_logger(logging.getLogger(__name__))
timer = Timer()


class ModelEvaluator:

    def __init__(self, y_classes):
        self.y_classes = y_classes

        self.names = []
        self.models = {}
        self.results = []
        self.confusion_matrices = []

    def _reset(self):
        self.names = []
        self.models = {}
        self.results = []
        self.confusion_matrices = []

    @staticmethod
    def _create_model_collection(n_estimators=10):
        return {
            "SVM": SVC(),
            "BAG": BaggingClassifier(n_estimators=n_estimators, n_jobs=-1),
            "RF": RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1),
            "ET": ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=-1)
        }

    def _eval_model(self, model, X, y) -> None:
        name = self.names[-1]
        logger.info(f"Step  - Evaluating {name} model BEGIN")
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        if type(y) is pd.DataFrame:
            y = y.to_numpy().ravel()
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

        self.results.append(scores)
        logger.info(f"Step  - Evaluating {name} model {timer.time_stage(f'{name} Evaluation')} END")

    def _get_class_for_label(self, label):
        return self.y_classes.get(label)

    def _get_all_classes_from_labels(self):
        return [self.y_classes.get(idx) for idx, label in enumerate(self.y_classes)]

    def run_model_evaluation(self, X_train, y_train, X_test, y_test):
        logger.info("Stage - Model Analysis BEGIN")
        self.models = self._create_model_collection(n_estimators=100)
        for idx, (name, model) in enumerate(self.models.items()):
            self.names.append(name)

            self._fit_model(X_train=X_train, y_train=y_train)

            y_pred = self._make_prediction(X_test=X_test)

            self._create_confusion_matrix(y_test=y_test, y_pred=y_pred)

        self._log_results()

        self.show_box_plots()

        logger.info("Stage - Model Analysis END")

    def _fit_model(self, X_train, y_train):
        name = self.names[-1]
        model = self.models[name]

        logger.info(f"Step  - Fitting {name} model BEGIN")
        model.fit(X_train, y_train.to_numpy().ravel())
        self.models[name] = model
        logger.info(f"Step  - Fitting {name} model END {timer.time_stage(f'Fitting Model {name}')}")

    def _make_prediction(self, X_test):
        name, model = self._name_and_model()

        logger.info(f"Step  - Prediction BEGIN")
        y_pred = model.predict(X_test).reshape(-1, 1)
        logger.info(f"Step  - Prediction END {timer.time_stage(f'y Prediction {name}')}")

        return y_pred

    def _create_confusion_matrix(self, y_test, y_pred):
        name, _ = self._name_and_model()

        logger.info(f"Step  - Confusion Matrix BEGIN")
        cm = confusion_matrix(y_test, y_pred)
        self.confusion_matrices.append(cm)
        logger.info(f"Step  - Confusion Matrix END {timer.time_stage(f'Confusion Matrix {name}')}")

    def _log_results(self):
        [logger.info(f"Step  - Results: {name} - mean={mean(scores):0.3f}, std={std(scores):0.3f}") for name, scores in
         zip(self.names, self.results)]

    def _name_and_model(self):
        name = self.names[-1]
        model = self.models[name]

        return name, model

    def show_box_plots(self):
        plt.boxplot(self.results, labels=list(self.models.keys()), showmeans=True)
        plt.show()

    def show_confusion_matrix(self):
        for name, cm in zip(self.names, self.confusion_matrices):
            div = np.sum(cm, axis=1)
            div_inv = div.reshape(-1, 1)
            cm_div = (cm / div_inv[None, :])
            cm_div = np.nan_to_num(cm_div, posinf=0.0, neginf=0.0)
            plt.subplots(figsize=(20, 10))
            ax: plt.Axes = sns.heatmap(cm_div[0], cmap="YlOrRd", linewidths=0.5, annot=True,
                                       xticklabels=self._get_all_classes_from_labels(),
                                       yticklabels=self._get_all_classes_from_labels(),
                                       fmt=".2f", cbar=False)
            ax.set_title(f"Confusion matrix ({name})")
            ax.set_xlabel("Predictions")
            ax.set_ylabel("Real Values")
            plt.show()
