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
    return 1 - (2*a.std())


def fairness_report(y, y_pred, sensitives, model_name):
    """Fairness report

    Args:
        y (array): Dataset Labels.
        y_pred (array): Model predictions.
        sensitives (dataframe): Test dataset of sensitive attributes.
        model_name (string): Name of model.

    Returns:
        Pandas dataframe: A row with all metrics and their values.
    """
    tn, fp, _, _ = confusion_matrix(y, y_pred).ravel()
    report = {
        "Accuracy": [accuracy_score(y, y_pred)],
        "Balanced Accuracy": [balanced_accuracy_score(y, y_pred)],
        "F1 Score": [f1_score(y, y_pred)],
        "Specificity": tn / (tn + fp),
    }

    # Calculate Independent Group Fairness
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
        name = "Parity Score " + sens
        report[name] = parity_score(likelihoods)

    # Calculate Intersection Group Fairness
    groupintersect = sensitives.groupby(
        sensitives.columns.tolist()
    ).size().reset_index().drop(0, axis=1)

    probs = groupintersect.apply(
        lambda row: y_pred[
                (sensitives == row.to_dict()).all(axis=1)
            ].value_counts(normalize=True),
        axis=1
        )[1].values.flatten()

    # Replace Nan
    probs = [0 if x != x else x for x in probs]
    report['Intersectional Parity Score'] = parity_score(probs)

    report["Model"] = model_name

    return pd.DataFrame(report)
