"""Bayesian Network Module."""

from pgmpy.models import BayesianNetwork, NaiveBayes
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import ExpectationMaximization as EM
from forseti.datproc import translate_categorical
import pickle
import numpy as np
from scipy.special import rel_entr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score


class latentLabelClassifier:
    """Latent Fair Label Bayesian Network Classifier."""

    def __init__(self, df, sensitives, label, atol=0.01):
        self.df = df
        self.sensitives = sensitives
        self.label = label
        self.atol = atol
        self.model = BayesianNetwork()
        self.classes = None
        self.codes = None

    def fit(self):
        """
        Constructs a bayesian network with a fair latent variable.

        Structure Learning: Hill Climb Search
        Parameter Learning: Expectation Maximation (EM)
        """
        tmp, codes = translate_categorical(self.df.copy(deep=True))
        self.classes = len(tmp[self.label].value_counts())
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
        self.model = pickle.load(open(file, "rb"))

    def save(self, file):
        """Save trained model to file using pickle.

        Args:
            file (filetype): _description_
        """
        pickle.dump(self.model, open(file, "wb"))

    def check_model(self):
        """Check if model is a valid model.

        Returns:
            _type_: boolean
        """
        return self.model.check_model()


class interpretableNaiveBayes(NaiveBayes):

    def train(self, label, df, name):
        self.name = name
        train, test = train_test_split(df, test_size=0.05)
        tmp_train, self.codes_train = translate_categorical(
            train.copy(deep=True)
        )
        self.X_test, _ = translate_categorical(test.copy(deep=True))
        self.y_test = self.X_test[label]
        self.X_test = self.X_test.drop(label, axis=1)
        self.fit(tmp_train, label)

    def KLDWeights(self):
        cpds = self.get_cpds()
        tmp = []

        for key, cpd in enumerate(cpds):
            if cpd.values.ndim == 2:
                cpd.values[cpd.values == 0] = 10e-3
                cpd.normalize()
                KLD = np.sum(rel_entr(cpd.values[:, 0], cpd.values[:, 1]))
                tmp.append(
                    [list(self.codes_train.keys())[key], KLD, self.name]
                )
            else:
                cpd.values[cpd.values == 0] = 10e-3
                cpd.normalize()
                KLD = np.sum(rel_entr(cpd.values[0], cpd.values[1]))
                tmp.append(
                    [list(self.codes_train.keys())[key], KLD, self.name]
                )

        df = pd.DataFrame(
            tmp,
            columns=['Attribute', 'KLD', 'Model']
        )

        df['KLD'] = df['KLD'].astype('float')

        return df

    def PermutationImportance(self, K, name):
        df = self.X_test
        y_pred = self.predict(df)
        s = balanced_accuracy_score(self.y_test, y_pred)
        Imp = []

        for col in self.X_test.columns:
            It = []
            for i in range(K):
                # Permute Column
                df[col] = np.random.permutation(df[col])
                yp = self.predict(df)
                It.append(balanced_accuracy_score(self.y_test, yp))
            Imp.append([col, s - np.mean(It), name])

        Imp = np.array(Imp)
        df = pd.DataFrame(
            Imp,
            columns=['Attribute', 'Weight', 'Model']
        )

        df['Weight'] = df['Weight'].astype('float')

        return df
