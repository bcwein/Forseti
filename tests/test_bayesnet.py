from forseti.bayesnet import latentLabelClassifier, interpretableNaiveBayes
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
        atol=0.01
    )

    clf.fit()
    assert clf.check_model()


def test_interpretableNaiveBayes():
    tmp = df[:10]
    label = 'income'

    clf = interpretableNaiveBayes()

    clf.train(
        label,
        tmp,
        'NB'
    )

    assert clf.check_model()
