"""Synthethic Dataset Generation Module."""

import pandas as pd
from scipy.stats import bernoulli, rv_discrete
import numpy as np
from scipy.stats import norm


def datasetgen_numerical(
    n_samples=1000,
    informative=True,
    seperability=0.5
):
    """Create Synthetic Datasets

    Returns:
        df: Pandas DataFrame.
    """
    df = pd.DataFrame()

    # Gender Attribute
    mapping_gender = {
        0: 'Male',
        1: 'Female'
    }
    df['Gender'] = [
        mapping_gender[x] for x in bernoulli.rvs(0.5, size=n_samples)
    ]
    df['Gender'] = df['Gender'].astype('category')

    # Race Attribute
    mapping_race = {
        0: 'Black',
        1: 'White',
        2: 'Hispanic',
        3: 'Asian',
        4: 'Other'
    }
    a = abs(np.random.randn(len(mapping_race)))
    a = a / a.sum()
    rv = rv_discrete(values=(range(len(a)), a))
    df['Race'] = [
        mapping_race[x] for x in rv.rvs(size=n_samples)
    ]
    df['Race'] = df['Race'].astype('category')

    # Gender
    lengths = df['Gender'].value_counts()

    df.loc[df['Gender'] == 'Male', 'Feature 1'] = norm(
        loc=1+seperability,
        scale=1
    ).rvs(lengths['Male'])

    df.loc[df['Gender'] == 'Female', 'Feature 1'] = norm(
        loc=1,
        scale=1
    ).rvs(lengths['Female'])

    df.loc[df['Gender'] == 'Male', 'Feature 2'] = norm(
        loc=1+seperability,
        scale=1
    ).rvs(lengths['Male'])

    df.loc[df['Gender'] == 'Female', 'Feature 2'] = norm(
        loc=1,
        scale=1
    ).rvs(lengths['Female'])

    # Race
    lengths = df['Race'].value_counts()

    # Feature 3
    for count, vals in enumerate(mapping_race.values()):
        df.loc[df['Race'] == vals, 'Feature 3'] = norm(
            loc=1 + count*seperability*np.random.choice((-1, 1)),
            scale=1
        ).rvs(lengths[vals])

    for count, vals in enumerate(mapping_race.values()):
        df.loc[df['Race'] == vals, 'Feature 4'] = norm(
            loc=2 + count*seperability*np.random.choice((-1, 1)),
            scale=1
        ).rvs(lengths[vals])

    # Informative Sensitives
    if informative:
        pass
    else:
        mapping_gender = {
            0: 'Male',
            1: 'Female'
        }
        df['Gender'] = [
            mapping_gender[x] for x in bernoulli.rvs(0.5, size=n_samples)
        ]
        df['Gender'] = df['Gender'].astype('category')

        # Race Attribute
        mapping_race = {
            0: 'Black',
            1: 'White',
            2: 'Hispanic',
            3: 'Asian',
            4: 'Other'
        }
        a = abs(np.random.randn(len(mapping_race)))
        a = a / a.sum()
        rv = rv_discrete(values=(range(len(a)), a))
        df['Race'] = [mapping_race[x] for x in rv.rvs(size=n_samples)]
        df['Race'] = df['Race'].astype('category')

    scores = df.drop(['Gender', 'Race'], axis=1).sum(axis=1)
    thr = scores.median()

    mapping_success = {
        0: 'No',
        1: 'Yes'
    }
    
    df['Success'] = [mapping_success[x] for x in (scores >= thr).astype('int')]

    return df
