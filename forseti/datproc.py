"""Data Processing Module."""

import pandas as pd


def translate_categorical(dataframe):
    """Takes a pandas dataframe and translate all categorical attributes to
    numerical values and returns encoded dataframe.

    Args:
        dataframe (pandas dataframe): Dataframe to translate
    """
    codes = {}

    # Translate object to categorical
    obje = dataframe.select_dtypes(["object"]).columns
    for obj in obje:
        dataframe[obj] = dataframe[obj].astype('category')

    # Translate numeric to categorical
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical = dataframe.select_dtypes(numerics).columns
    for num in numerical:
        dataframe[num] = pd.cut(dataframe[num], 5, duplicates='drop')
        dataframe[num] = dataframe[num].astype('category')

    # Enumerate categorical
    categorical = dataframe.select_dtypes(["category"]).columns

    for cat in categorical:
        dataframe[cat] = dataframe[cat].astype("category")
        codes[cat] = dict(enumerate(dataframe[cat].cat.categories))
        dataframe[cat] = dataframe[cat].cat.codes
        dataframe[cat] = dataframe[cat].astype('category')

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
