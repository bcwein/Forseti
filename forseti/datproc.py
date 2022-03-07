"""Data Processing Module."""

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
