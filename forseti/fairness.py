"""Module for performance metrics."""

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
import pandas as pd
import numpy as np


def parity_score(probabilities):
    """Parity Score Function.

    It calculates the parity score from a list of probabilities.
    These probabilities are assumed to be conditional probabilities on the form
    P(Y|S_0), P(Y|S_1), ... P(Y|S_n). I.e probability of outcome conditional on
    a sensitive class.

    The scoring function transform a list of probabilities to a number between
    0 and 1. Where 0 is the worst possible score (where there is at least one
    probability of 0) and the best score where all probabilities are equal.

    For a given list of probabilities L, the score is calculated as

    Score = geometric mean(L) / mean(Ls)

    Args:
        probabilities (list): list of sensitive conditional probabilities.

    Returns:
        float: A score between 0 and 1.
    """
    a = np.array(probabilities)
    return a.prod() ** (1.0 / len(a)) / a.mean()


def fairness_report(y, y_pred, sensitives, model_name):
    tn, fp, _, _ = confusion_matrix(y, y_pred).ravel()
    report = {
        "Accuracy": [accuracy_score(y, y_pred)],
        "Balanced Accuracy": [balanced_accuracy_score(y, y_pred)],
        "F1 Score": [f1_score(y, y_pred)],
        "Specificity": tn / (tn + fp),
    }

    for sens in sensitives.columns:
        likelihoods = []
        for val in sorted(sensitives[sens].unique()):
            try:
                likelihoods.append(
                    y_pred[sensitives[sens] == val].value_counts(
                        normalize=True
                    )[1]
                )
            except KeyError:
                likelihoods.append(0)
        name = "Positive Parity Score " + sens
        report[name] = parity_score(likelihoods)

    report["Model"] = model_name

    return pd.DataFrame(report)
