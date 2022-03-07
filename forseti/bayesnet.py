"""Bayesian Network Module."""

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import ExpectationMaximization as EM
from pgmpy.base import DAG
from forseti.datproc import translate_categorical


def latent_label_classifier(df, sensitives, label, atol=0.01):
    tmp, codes = translate_categorical(df.copy(deep=True))
    tmp = tmp.drop(label, axis=1)

    sensitive = ['gender', 'race']
    blacklist = []
    for col in tmp.columns:
        blacklist.append((col, sensitive[0]))
        blacklist.append((col, sensitive[1]))

    c = HillClimbSearch(tmp)
    model = c.estimate(black_list=blacklist)

    for sens in sensitives:
        model.add_edge(sens, label)

    tmp, codes = translate_categorical(df.copy(deep=True))

    # Connect latent to features
    for col in tmp.columns:
        if col in sensitives:
            continue
        else:
            model.add_edge('fair', col)

    # Connect latent to label
    model.add_edge('fair', label)

    fair_model = BayesianNetwork(list(model.edges()), latents={'fair'})

    estimator = EM(fair_model, tmp)
    cpds = estimator.get_parameters(latent_card={'fair': 2}, atol=atol)

    for cpd in cpds:
        fair_model.add_cpds(cpd)

    return fair_model, codes
