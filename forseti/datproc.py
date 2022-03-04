"""Data Processing Module."""

import numpy as np
import pandas as pd


def translate_categorical(dataframe):
    """Takes a pandas dataframe and translate all categorical attributes to
    numerical values.

    Args:
        dataframe (pandas dataframe): Dataframe to translate
    """
    codes = {}
    categorical = dataframe.select_dtypes(["object"]).columns

    for cat in categorical:
        dataframe[cat] = dataframe[cat].astype("category")
        codes[cat] = dict(enumerate(dataframe[cat].cat.categories))
        dataframe[cat] = dataframe[cat].cat.codes

    return dataframe, codes


def extract_sensitive(df, attributes):
    """Takes a pandas dataframe and extract sensitive attributes from list of
    attributes.

    Args:
        dataframe (pandas dataframe): Dataframe to translate
        attributes: sensitive attributes
    Return:
        sensitive: Dataframe of sensitive labels
        features: Non-sensitive features
    """
    sensitive = df[attributes]
    features = df.drop(attributes, axis=1)
    return sensitive, features


def encode_dummies(df):
    dummy = pd.get_dummies(df, prefix_sep=".", drop_first=True)
    return dummy


def mean(x, w=1):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)


def cov(x, y, w=1):
    """Weighted Covariance"""
    return np.sum(w * (x - mean(x, w)) * (y - mean(y, w))) / np.sum(w)


def corr(x, y, w=1):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))


def weighted_corr(df, weight, target):
    """Weighted correlation on target variable in dataset.

    Args:
        df (Pandas Dataframe): Data
        weight (String): Name of the weight column in the dataframe
        target (String): Name of the target variable.

    Returns:
        Dataframe: Dataframe of correlations.
    """
    w = df[weight]
    y = df[target]
    columns = df.drop([weight, target], axis=1).columns
    r = {}

    for col in columns:
        r[col] = corr(df[col], y, w)

    return pd.DataFrame.from_dict(
        r, orient="index", columns=["Correlation"]
    ).sort_values(by="Correlation")
