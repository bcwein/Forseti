"""Bayesian Network Module."""

from pgmpy.models import BayesianNetwork, NaiveBayes
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import ExpectationMaximization as EM
from forseti.datproc import translate_categorical
import pickle
import numpy as np
from scipy.special import rel_entr
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt


class latentLabelClassifier:
    """Latent Fair Label Bayesian Network Classifier."""

    def __init__(self, df, sensitives, label, atol=0.01):
        train, test = train_test_split(df, test_size=0.33)
        self.train, self.codes = translate_categorical(train)
        self.test, _ = translate_categorical(test)
        self.y = self.test[label]
        self.test = self.test.drop(label, axis=1)
        self.sensitives = sensitives
        self.label = label
        self.atol = atol
        self.model = BayesianNetwork()
        self.classes = None

    def fit(self):
        """
        Constructs a bayesian network with a fair latent variable.

        Structure Learning: Hill Climb Search
        Parameter Learning: Expectation Maximation (EM)
        """
        self.classes = len(self.train[self.label].value_counts())

        blacklist = []
        for col in self.train.drop(self.label, axis=1):
            blacklist.append((col, self.sensitives[0]))
            blacklist.append((col, self.sensitives[1]))

        c = HillClimbSearch(self.train.drop(self.label, axis=1))
        model = c.estimate(black_list=blacklist)

        for sens in self.sensitives:
            model.add_edge(sens, self.label)

        # Connect latent to features
        for col in self.train.drop(self.label, axis=1):
            if col in self.sensitives:
                continue
            else:
                model.add_edge("fair", col)

        # Connect latent to label
        model.add_edge("fair", self.label)

        fair_model = BayesianNetwork(list(model.edges()), latents={"fair"})

        estimator = EM(fair_model, self.train)
        cpds = estimator.get_parameters(
            latent_card={"fair": self.classes}, atol=self.atol
        )

        for cpd in cpds:
            fair_model.add_cpds(cpd)

        self.model = fair_model

    def predict_probability(self):
        """Predict and return probabilities.

        Return:
            dataframe: Dataframe with predictions for unobserved variables.
        """
        return self.model.predict_probability(self.test)

    def predict(self):
        """Predict and return prediction labels.

        Return:
            dataframe: Dataframe with predictions for unobserved variables.
        """
        return self.model.predict(self.test)

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

    def PermutationImportance(self, K, name):
        """Permutation Feature Importance.

        Estimate feature importance by permutating column of interest and
        calculate loss of score.

        Args:
            K: No of permutation iterations.
            name: Name of model used.

        Returns:
            df: Pandas dataframe of feature importance.
        """
        df = self.test.copy(deep=True)
        y_pred = self.model.predict(df)['fair']
        s = balanced_accuracy_score(self.y, y_pred)
        Imp = []

        for col in self.test.columns:
            It = []
            for i in range(K):
                # Permute Column
                df[col] = np.random.permutation(df[col])
                yp = self.model.predict(df)['fair']
                It.append(balanced_accuracy_score(self.y, yp))
            Imp.append([col, s - np.mean(It), name])

        Imp = np.array(Imp)
        df = pd.DataFrame(
            Imp,
            columns=['Attribute', 'Weight', 'Model']
        )

        df['Weight'] = df['Weight'].astype('float')

        return df

    def ICEPlot(self, attr, samples=100):
        df = pd.DataFrame()
        a = self.test.sample(samples)

        for val in self.codes[attr]:
            a[attr] = val
            tmp = self.model.predict_probability(a)
            tmp[attr] = val
            df = df.append(tmp)

        if isinstance(self.codes[attr][0], pd._libs.interval.Interval):
            pass
        else:
            df = df.replace({attr: self.codes[attr]})

        for idx in df.index.unique():
            plt.plot(
                df.loc[idx][attr],
                df.loc[idx].iloc[:, 1],
                c='#1f77b4',
                alpha=0.5
            )

        plt.xlabel(attr)
        plt.ylabel('Positive Outcome Probability')
        plt.xticks(rotation=45)


class interpretableNaiveBayes(NaiveBayes):
    """Extension of Naive Bayes in pgmpy."""
    def train(self, label, df, name):
        """Fit Naive Bayes model on data.

        Args:
            label: Target variable.
            df: Pandas dataframe of data.
            name: Name given to the NB model.
        """
        self.label = label
        self.name = name
        train, test = train_test_split(df, test_size=0.05)
        self.train_train, self.codes_train = translate_categorical(
            train.copy(deep=True)
        )
        self.X_test, _ = translate_categorical(test.copy(deep=True))
        self.y_test = self.X_test[label]
        self.X_test = self.X_test.drop(label, axis=1)
        self.fit(self.train_train, label)

    def KLDWeights(self):
        """KLD Weights.

        Calculate attribute weights using KLD on conditional probabilities.

        Returns:
            df: Pandas dataframe of weights.
        """
        cpds = self.get_cpds()
        self.train = []

        for attr, cpd in enumerate(cpds):
            if cpd.values.ndim == 2:
                cpd.values[cpd.values == 0] = 10e-3
                cpd.normalize()
                KLD = np.sum(rel_entr(cpd.values[:, 0], cpd.values[:, 1]))
                self.train.append(
                    [list(self.codes_train.attrs())[attr], KLD, self.name]
                )
            else:
                cpd.values[cpd.values == 0] = 10e-3
                cpd.normalize()
                KLD = np.sum(rel_entr(cpd.values[0], cpd.values[1]))
                self.train.append(
                    [list(self.codes_train.attrs())[attr], KLD, self.name]
                )

        df = pd.DataFrame(
            self.train,
            columns=['Attribute', 'KLD', 'Model']
        )

        df['KLD'] = df['KLD'].astype('float')

        return df

    def PermutationImportance(self, K, name):
        """Permutation Feature Importance.

        Estimate feature importance by permutating column of interest and
        calculate loss of score.

        Args:
            K: No of permutation iterations.
            name: Name of model used.

        Returns:
            df: Pandas dataframe of feature importance.
        """
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

    def ICEplot(self, attr, samples=100):
        df = pd.DataFrame()
        a = self.X_test.sample(samples)

        for val in self.codes_train[attr]:
            a[attr] = val
            tmp = self.predict_probability(a)
            tmp[attr] = val
            df = df.append(tmp)

        if isinstance(self.codes_train[attr][0], pd._libs.interval.Interval):
            pass
        else:
            df = df.replace({attr: self.codes_train[attr]})

        for idx in df.index.unique():
            plt.plot(
                df.loc[idx][attr],
                df.loc[idx].iloc[:, 1],
                c='#1f77b4',
                alpha=0.5
            )

        plt.xlabel(attr)
        plt.ylabel('Positive Outcome Probability')
        plt.xticks(rotation=45)

    def generateCounterfactuals(self, candidates=100):

        datapoint = self.X_test.sample(1)

        # Find candidate that has negative prediction
        while self.predict_probability(datapoint).iloc[0, 1] >= 0.5:
            datapoint = self.X_test.sample(1)

        # Sample Candidates
        X = datapoint.sample(candidates, replace=True)

        def randomChange(row):
            col = random.choice(X.columns)
            val = random.choice(list(self.codes_train[col].keys()))
            row[col] = val
            return row

        X = X.apply(lambda row: randomChange(row), axis=1)

        # Objective 1
        o1 = np.absolute(self.predict_probability(X).iloc[:, 1] - 1)

        # Objective 2
        o2 = X.apply(
            lambda row: (row == datapoint).iloc[0], axis=1
        ).astype('int').mean(axis=1)

        # Objective 3
        equaldf = X.apply(
            lambda row: (row == datapoint).iloc[0], axis=1
        ).astype('int')
        o3 = equaldf[equaldf == 0].count(axis=1)

        # Objective 4
        Xobs = self.train_train.sample(100).drop(self.label, axis=1)

        def getClosest(datapoint):
            return Xobs.apply(
                lambda row: (row == datapoint), axis=1
            ).astype('int').mean(axis=1).sort_values().iloc[0]

        o4 = X.apply(
            lambda row: getClosest(row),
            axis=1
        ).sort_values()

        X['O1'] = o1
        X['O2'] = o2
        X['O3'] = o3
        X['O4'] = o4

        return X
