"""Bayesian Network Module."""

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import ExpectationMaximization as EM
from forseti.datproc import translate_categorical
import pickle


class latentLabelClassifier:
    """Latent Fair Label Bayesian Network Classifier."""

    def __init__(self, df, sensitives, label, atol=0.01, classes=2):
        self.df = df
        self.sensitives = sensitives
        self.label = label
        self.atol = atol
        self.classes = classes
        self.model = BayesianNetwork()
        self.codes = None

    def fit(self):
        """
        Constructs a bayesian network with a fair latent variable.

        Structure Learning: Hill Climb Search
        Parameter Learning: Expectation Maximation (EM)
        """
        tmp, codes = translate_categorical(self.df.copy(deep=True))
        tmp = tmp.drop(self.label, axis=1)

        blacklist = []
        for col in tmp.columns:
            blacklist.append((col, self.sensitives[0]))
            blacklist.append((col, self.sensitives[1]))

        c = HillClimbSearch(tmp)
        model = c.estimate(black_list=blacklist)

        for sens in self.sensitives:
            model.add_edge(sens, self.label)

        tmp, codes = translate_categorical(self.df.copy(deep=True))

        # Connect latent to features
        for col in tmp.columns:
            if col in self.sensitives:
                continue
            else:
                model.add_edge("fair", col)

        # Connect latent to label
        model.add_edge("fair", self.label)

        fair_model = BayesianNetwork(list(model.edges()), latents={"fair"})

        estimator = EM(fair_model, tmp)
        cpds = estimator.get_parameters(
            latent_card={"fair": self.classes}, atol=self.atol
        )

        for cpd in cpds:
            fair_model.add_cpds(cpd)

        self.model = fair_model
        self.codes = codes

    def predict_probability(self, test):
        """Predict and return probabilities.

        Args:
            test (Dataframe): Test Dataset

        Return:
            dataframe: Dataframe with predictions for unobserved variables.
        """
        return self.model.predict_probability(test)

    def predict(self, test):
        """Predict and return prediction labels.

        Args:
            test (Dataframe): Test Dataset

        Return:
            dataframe: Dataframe with predictions for unobserved variables.
        """
        return self.model.predict(test)

    def load(self, file):
        """Load trained model from file using pickle.

        Args:
            file (string): file path of trained model.
        """
        self.model = pickle.load(open(file, 'rb'))

    def save(self, file):
        """Save trained model to file using pickle.

        Args:
            file (filetype): _description_
        """
        pickle.dump(self.model, open(file, 'wb'))
