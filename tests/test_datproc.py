from forseti.datproc import translate_categorical, extract_sensitive
import pandas as pd

df = pd.read_csv('notebooks/data/adult.csv')


def test_translate_categorical():
    tmp, codes = translate_categorical(df)
    assert type(tmp) is pd.core.frame.DataFrame
    assert type(codes) is dict


def test_extract_sensitive():
    sensattr = ['gender', 'race']
    sensitive, features = extract_sensitive(df, sensattr)
    pd.testing.assert_frame_equal(df[sensattr], sensitive)
    pd.testing.assert_frame_equal(df.drop(sensattr, axis=1), features)
