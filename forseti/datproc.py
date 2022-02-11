"""Data Processing Module."""


def translate_categorical(dataframe):
    """Takes a pandas dataframe and translate all categorical attributes to
    numerical values.

    Args:
        dataframe (pandas dataframe): Dataframe to translate
    """
    codes = {}
    categorical = dataframe.select_dtypes(['object']).columns

    for cat in categorical:
        dataframe[cat] = dataframe[cat].astype('category')
        codes[cat] = dict(enumerate(dataframe[cat].cat.categories))
        dataframe[cat] = dataframe[cat].cat.codes

    return dataframe, codes