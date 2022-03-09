from forseti.bayesnet import latentLabelClassifier
import pandas as pd

df = pd.read_csv('notebooks/data/adult.csv')


def test_latentLabelClassifier():
    # Test on small subsample of dataset
    tmp = df[:10]
    sensitives = ['gender', 'race']
    label = 'income'

    clf = latentLabelClassifier(
        tmp,
        sensitives,
        label,
        atol=0.01,
        classes=2
    )

    clf.fit()
    assert clf.check_model()
