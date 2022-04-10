"""Synthethic Dataset Generation Module."""

import pandas as pd
from scipy.stats import bernoulli, rv_discrete
import numpy as np


def datasetgen(
    n_samples=1000
):
    """Create Synthetic Datasets

    Returns:
        df: Pandas DataFrame.
    """
    df = pd.DataFrame()

    # Gender Attribute
    mapping = {
        0: 'Male',
        1: 'Female'
    }
    df['Gender'] = [mapping[x] for x in bernoulli.rvs(0.5, size=n_samples)]

    # Race Attribute
    mapping = {
        0: 'Black',
        1: 'White',
        2: 'Hispanic',
        3: 'Asian',
        4: 'Other'
    }
    a = abs(np.random.randn(len(mapping)))
    a = a / a.sum()
    rv = rv_discrete(values=(range(len(a)), a))
    df['Race'] = [mapping[x] for x in rv.rvs(size=n_samples)]

    return df
