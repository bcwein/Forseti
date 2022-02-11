from forseti.datproc import translate_categorical
import pandas as pd


def test_translate_categorical():
    df = pd.DataFrame()
    assert type(translate_categorical(df)) == "<class 'pandas.core.frame.DataFrame'>"
