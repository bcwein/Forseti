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

    def ICE(self, attr, samples=100):
        df = pd.DataFrame()
        a = self.test.sample(samples)
        for val in self.codes[attr]:
            a[attr] = val
            tmp = self.model.predict_probability(a).iloc[:, -2:]
            tmp.columns = ['negative_outcome', 'positive_outcome']
            tmp[attr] = val
            df = df.append(tmp)

        if isinstance(self.codes[attr][0], pd._libs.interval.Interval):
            pass
        else:
            df = df.replace({attr: self.codes[attr]})

        return df

    def generateCounterfactuals(self, datapoint, candidates=100, gen=1):

        """Helper Functions"""
        def randomChange(row, mutation):
            for i in range(mutation):
                col = random.choice(P.columns)
                val = random.choice(list(self.codes[col].keys()))
                row[col] = val
            return row

        def calculateObjective(data):
            # Objective 1
            o1 = np.absolute(
                self.model.predict_probability(data).iloc[:, 1] - 1
            )

            # Objective 2
            o2 = data.apply(
                lambda row: (row == datapoint).iloc[0], axis=1
            ).astype('int').mean(axis=1)

            # Objective 3
            equaldf = data.apply(
                lambda row: (row == datapoint).iloc[0], axis=1
            ).astype('int')
            o3 = equaldf[equaldf == 0].count(axis=1)

            # Objective 4
            Xobs = self.train.sample(100).drop(self.label, axis=1)

            def getClosest(datapoint):
                return Xobs.apply(
                    lambda row: (row == datapoint), axis=1
                ).astype('int').mean(axis=1).sort_values().iloc[0]

            o4 = data.apply(
                lambda row: getClosest(row),
                axis=1
            ).sort_values()

            data['O1'] = o1
            data['O2'] = o2
            data['O3'] = o3
            data['O4'] = o4

            data = data.reset_index(drop=True)
            data = data.fillna(1.0)
            return data

        def dominates(row1, row2, P):
            opt = ['O1', 'O2', 'O3', 'O4']
            return (P.iloc[row1][opt] <= P.iloc[row2][opt]).sum() < 4

        def recombination(parents, children=100):
            Q = pd.DataFrame()
            df = P.iloc[parents]
            for i in range(children):
                child = df.iloc[0].copy()
                for col in df.columns:
                    child[col] = df[col].sample(1)
                # Mutate Child
                child = randomChange(child, 1)
                Q = Q.append(child)
            Q = Q.reset_index(drop=True)
            return Q

        def crowdDistance(F1, df):
            In = df.loc[list(F1)]

            In.insert(len(In.columns), 'Distance', 0)

            for obj in ['O1', 'O2', 'O3', 'O4']:
                In = In.sort_values(obj)
                In.iloc[0, In.columns.get_loc('Distance')] = np.inf
                In.iloc[-1, In.columns.get_loc('Distance')] = np.inf
                for i in range(len(In) - 1):
                    In.iloc[i, In.columns.get_loc('Distance')] = \
                        In.iloc[i, In.columns.get_loc('Distance')] + \
                        In.iloc[i+1][obj] - In.iloc[i-1][obj] / \
                        abs(In[obj].max() - In[obj].min())

            In = In.sort_values('Distance', ascending=False)
            return In

        """START OF COUNTERFACTUAL GENERATION!"""
        # Sample Candidates
        P = datapoint.sample(candidates, replace=True)

        # Generate random parent population
        P = P.apply(
            lambda row: randomChange(
                    row, random.randint(0, int(len(row)))
                ), axis=1
            )

        # Calculate Objectives for Population
        P = calculateObjective(P)

        # Rank parent population
        F1 = set()
        for p in range(len(P)):
            Sp = set()
            Np = 0
            for q in range(len(P)):
                if dominates(p, q, P):
                    Sp.add(q)
                elif dominates(q, p, P):
                    Np = Np + 1
            if Np == 0:
                F1.add(p)

        Pcrowd = crowdDistance(F1, P)
        parents = [Pcrowd.index[0], Pcrowd.index[1]]
        P = P.drop(['O1', 'O2', 'O3', 'O4'], axis=1)
        Q = recombination(parents)
        R = P.append(Q, ignore_index=True)
        R = R.apply(lambda x: x.astype('int'))
        R = calculateObjective(R)

        # Repeat N number of generations
        for i in range(gen):
            # Rank population
            F1 = set()
            F = {}
            N = np.zeros(len(R))
            S = {}
            for p in range(len(R)):
                S[p] = set()
                for q in range(len(R)):
                    if dominates(p, q, R):
                        S[p].add(q)
                    elif dominates(q, p, R):
                        N[p] = N[p] + 1
                if N[p] == 0:
                    F1.add(p)
            i = 0
            F[i] = F1
            while F[i]:
                Q = set()
                for p in F[i]:
                    for q in S[p]:
                        N[q] = N[q] - 1
                        if N[q] == 0:
                            Q.add(q)
                i = i + 1
                F[i] = Q

            Rnew = set()
            i = 0
            while len(Rnew) + len(F[i]) < 100:
                Rnew = Rnew.union(F[i])
                i = i + 1

            Franked = crowdDistance(F[i], R)
            num = (100 - len(Rnew))
            Rnew = Rnew.union(list(Franked.iloc[:num].index))
            P = R.loc[Rnew]
            P = P.drop(['O1', 'O2', 'O3', 'O4'], axis=1)
            Q = recombination(parents)
            R = P.append(Q, ignore_index=True)
            R = R.apply(lambda x: x.astype('int'))
            R = calculateObjective(R)

        R = R.replace(self.codes)
        datapoint = datapoint.replace(self.codes)
        return datapoint, R


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

    def ICE(self, attr, samples=100):
        df = pd.DataFrame()
        a = self.X_test.sample(samples)
        for val in self.codes_train[attr]:
            a[attr] = val
            tmp = self.predict_probability(a)
            tmp.columns = ['negative_outcome', 'positive_outcome']
            tmp[attr] = val
            df = df.append(tmp)

        if isinstance(self.codes_train[attr][0], pd._libs.interval.Interval):
            pass
        else:
            df = df.replace({attr: self.codes_train[attr]})

        return df

    def generateCounterfactuals(self, datapoint, candidates=100, gen=1):

        """Helper Functions"""
        def randomChange(row, mutation):
            for i in range(mutation):
                col = random.choice(P.columns)
                val = random.choice(list(self.codes_train[col].keys()))
                row[col] = val
            return row

        def calculateObjective(data):
            # Objective 1
            o1 = np.absolute(self.predict_probability(data).iloc[:, 1] - 1)

            # Objective 2
            o2 = data.apply(
                lambda row: (row == datapoint).iloc[0], axis=1
            ).astype('int').mean(axis=1)

            # Objective 3
            equaldf = data.apply(
                lambda row: (row == datapoint).iloc[0], axis=1
            ).astype('int')
            o3 = equaldf[equaldf == 0].count(axis=1)

            # Objective 4
            Xobs = self.train_train.sample(100).drop(self.label, axis=1)

            def getClosest(datapoint):
                return Xobs.apply(
                    lambda row: (row == datapoint), axis=1
                ).astype('int').mean(axis=1).sort_values().iloc[0]

            o4 = data.apply(
                lambda row: getClosest(row),
                axis=1
            ).sort_values()

            data['O1'] = o1
            data['O2'] = o2
            data['O3'] = o3
            data['O4'] = o4

            data = data.reset_index(drop=True)
            data = data.fillna(1.0)
            return data

        def dominates(row1, row2, P):
            opt = ['O1', 'O2', 'O3', 'O4']
            return (P.iloc[row1][opt] <= P.iloc[row2][opt]).sum() < 4

        def recombination(parents, children=100):
            Q = pd.DataFrame()
            df = P.iloc[parents]
            for i in range(children):
                child = df.iloc[0].copy()
                for col in df.columns:
                    child[col] = df[col].sample(1)
                # Mutate Child
                child = randomChange(child, 1)
                Q = Q.append(child)
            Q = Q.reset_index(drop=True)
            return Q

        def crowdDistance(F1, df):
            In = df.loc[list(F1)]

            In.insert(len(In.columns), 'Distance', 0)

            for obj in ['O1', 'O2', 'O3', 'O4']:
                In = In.sort_values(obj)
                In.iloc[0, In.columns.get_loc('Distance')] = np.inf
                In.iloc[-1, In.columns.get_loc('Distance')] = np.inf
                for i in range(len(In) - 1):
                    In.iloc[i, In.columns.get_loc('Distance')] = \
                        In.iloc[i, In.columns.get_loc('Distance')] + \
                        In.iloc[i+1][obj] - In.iloc[i-1][obj] / \
                        abs(In[obj].max() - In[obj].min())

            In = In.sort_values('Distance', ascending=False)
            return In

        """START OF COUNTERFACTUAL GENERATION!"""
        # Sample Candidates
        P = datapoint.sample(candidates, replace=True)

        # Generate random parent population
        P = P.apply(
            lambda row: randomChange(
                    row, random.randint(0, int(len(row)))
                ), axis=1
            )

        # Calculate Objectives for Population
        P = calculateObjective(P)

        # Rank parent population
        F1 = set()
        for p in range(len(P)):
            Sp = set()
            Np = 0
            for q in range(len(P)):
                if dominates(p, q, P):
                    Sp.add(q)
                elif dominates(q, p, P):
                    Np = Np + 1
            if Np == 0:
                F1.add(p)

        Pcrowd = crowdDistance(F1, P)
        parents = [Pcrowd.index[0], Pcrowd.index[1]]
        P = P.drop(['O1', 'O2', 'O3', 'O4'], axis=1)
        Q = recombination(parents)
        R = P.append(Q, ignore_index=True)
        R = R.apply(lambda x: x.astype('int'))
        R = calculateObjective(R)

        # Repeat N number of generations
        for i in range(gen):
            # Rank population
            F1 = set()
            F = {}
            N = np.zeros(len(R))
            S = {}
            for p in range(len(R)):
                S[p] = set()
                for q in range(len(R)):
                    if dominates(p, q, R):
                        S[p].add(q)
                    elif dominates(q, p, R):
                        N[p] = N[p] + 1
                if N[p] == 0:
                    F1.add(p)
            i = 0
            F[i] = F1
            while F[i]:
                Q = set()
                for p in F[i]:
                    for q in S[p]:
                        N[q] = N[q] - 1
                        if N[q] == 0:
                            Q.add(q)
                i = i + 1
                F[i] = Q

            Rnew = set()
            i = 0
            while len(Rnew) + len(F[i]) < 100:
                Rnew = Rnew.union(F[i])
                i = i + 1

            Franked = crowdDistance(F[i], R)
            num = (100 - len(Rnew))
            Rnew = Rnew.union(list(Franked.iloc[:num].index))
            P = R.loc[Rnew]
            P = P.drop(['O1', 'O2', 'O3', 'O4'], axis=1)
            Q = recombination(parents)
            R = P.append(Q, ignore_index=True)
            R = R.apply(lambda x: x.astype('int'))
            R = calculateObjective(R)

        R = R.replace(self.codes_train)
        datapoint = datapoint.replace(self.codes_train)
        return datapoint, R
