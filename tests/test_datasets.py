from forseti.datasets import datasetgen
import pandas as pd


def test_datasetgen():
    df = datasetgen(
        n_samples=10
    )
    assert isinstance(df, pd.DataFrame)
    assert 'Gender' in df
    assert 'Race' in df
    assert len(df.index) == 10
