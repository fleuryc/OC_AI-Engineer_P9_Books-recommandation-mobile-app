"""Helper functions, not project specific."""
import warnings
from time import time
from typing import Any, Union

import pandas as pd
from sklearn.base import ClassifierMixin, is_classifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401,W0611
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold

warnings.filterwarnings(action="ignore", category=UserWarning)


def find_best_params_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    estimator: ClassifierMixin,
    params: dict[str, list[Union[str, float, int, bool]]] = None,
) -> dict[str, Any]:
    """Runs cross validation to find the best hyper-parameters of estimator.

    Args:
        X_train (pd.DataFrame): training data
        y_train (pd.Series): training labels
        X_test (pd.DataFrame): testing data
        y_test (pd.Series): testing labels
        estimator (ClassifierMixin): Classifier
        params (dict[str, list[Union[str, float, int, bool]]], optional):
            hyper-parameters range for cross validation. Defaults to {}.

    Raises:
        ValueError: Error if estimator is not a classifier

    Returns:
        dict[str, Any]: Classifier optimization results.
    """
    if not is_classifier(estimator):
        raise ValueError(f"{estimator} is not a classifier.")

    clf = HalvingRandomSearchCV(
        estimator=estimator,
        param_distributions=params,
        # StratifiedKFold Cross Validator
        # StratifiedKFold permet de séparer les données en nombre de folds de
        # manière stratifiée. Les proportions des classes sont conservées.
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        # F1 Score
        # F1 Score permet de mesurer la qualité d'un modèle en évaluant
        # la précision et le recall.
        scoring="f1",
        verbose=0,
        n_jobs=-1,
        random_state=42,
    ).fit(
        X=X_train,
        y=y_train,
    )

    start_time = time()
    y_pred = clf.predict(X_test)
    predict_time = time() - start_time

    if hasattr(clf, "predict_proba"):
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_pred_proba = clf.decision_function(X_test)
    else:
        y_pred_proba = y_pred

    return {
        "classifier": clf,
        "model": clf.best_estimator_,
        "params": clf.best_params_,
        "score": clf.best_score_,
        "predict_time": predict_time,
        "cv_results_": clf.cv_results_,
        "best_index_": clf.best_index_,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "average_precision": average_precision_score(y_test, y_pred_proba),
        "precision_recall_curve": precision_recall_curve(y_test, y_pred_proba),
        "roc_auc_score": roc_auc_score(y_test, y_pred_proba),
        "roc_curve": roc_curve(y_test, y_pred_proba),
    }


def automl_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    estimator: ClassifierMixin,
) -> dict[str, Any]:
    """
    Runs AutoML to find the best estimator.

    Args:
        X_train (pd.DataFrame): training data
        y_train (pd.Series): training labels
        X_test (pd.DataFrame): testing data
        y_test (pd.Series): testing labels
        estimator (ClassifierMixin): AutoML estimator

    Raises:
        ValueError: Error if estimator is not a classifier

    Returns:
        dict[str, Any]: Classifier optimization results.
    """
    if not is_classifier(estimator):
        raise ValueError(f"{estimator} is not a classifier.")

    clf = estimator.fit(X_train, y_train)

    start_time = time()
    y_pred = clf.predict(X_test)
    predict_time = time() - start_time

    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    return {
        "model": clf,
        "params": clf.get_params(),
        "score": float(pd.DataFrame(clf.cv_results_)[["mean_test_score"]].max()),
        "predict_time": predict_time,
        "cv_results_": clf.cv_results_,
        "best_index_": int(pd.DataFrame(clf.cv_results_)[["mean_test_score"]].idxmax()),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "average_precision": average_precision_score(y_test, y_pred_proba),
        "precision_recall_curve": precision_recall_curve(y_test, y_pred_proba),
        "roc_auc_score": roc_auc_score(y_test, y_pred_proba),
        "roc_curve": roc_curve(y_test, y_pred_proba),
    }
