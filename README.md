# Forseti
Fair machine learning python package.

This python package implements machine learning methods and functionality for evaluating machine learning models in terms of fairness.  See sections below for each module and classes.

# Modules

## bayesnet
Module for running bayesian networks 

<details>
<summary>Click here for documentation</summary>

### class **latentLabelClassifier**

Bayesian network which constructs a bayesian network that models the discrimination process. It assumes that the labels in the training dataset is biased and is genereted from a probability distribution

$$
P(D | D_f, S)
$$

where $D$ is the dataset labels, $D_f$ is the fair and unobservable true labels and $S$ are the sensitive attributes. The model learns a bayesian network for the joint probability

$$
P(D, D_f, S, X) = P(D|D_f, S)P(X|D_f, S)P(D_f)P(S)
$$

where $X$ is the non-sensitive features of the dataset. 
For learning the model structure, the **Hill Climbing** algorithm is used and for learning the model parameters **Expectation Maximiation** is used. There are implemented in [pgmpy](https://pgmpy.org/index.html) and is the implementations used here.

Prediction is done by estimating

$$
P(D_f | X, S)
$$

>latentLabelClassifier()

**Description:** Constructor for a new latentLabelClassifier

**Parameters:** 
- df: Training dataset 
- sensitives: List of sensitive attributes (must be in df)
- label: Dataset labels (must be in df)
- atol: Accepted Tolerance for expectation maximization
- classes: No of classes in the labels

>fit()

**Description:** Constructs the bayesian network with a fair latent variable.

Structure Learning: Hill Climb Search

Parameter Learning: Expectation Maximation (EM)

>predict_probability()

**Description:** Predict and return probabilities for unobserved nodes.

**Parameters:**
- test: Test dataset (without data labels and with sensitive attributes.)

>predict()

**Description:** Predict and return labels for unobserved nodes.

**Parameters:**
- test: Test dataset (without data labels and with sensitive attributes.)

>load()

**Description:** Load a learned model from pickle file. See save() for file 
format.

**Parameters:**
- file: File path of learned model.

>save()

**Description:** Save a learned model using pickle.

```
pickle.dump(self.model, open(file, "wb"))
```
**Parameters:**
- file: File path of learned model.

>check_model()

**Description:** Checks if model is valid. Returns true or false.
</details>

## datproc

Module for data preprocessing.

<details>
<summary>Click here for documentation</summary>

>translate_categorical()

**Description:** Takes a pandas dataframe and translate all categorical 
attributes to numerical values and returns encoded dataframe.

**Parameters:**
- dataframe (pandas dataframe): Dataframe to translate

>extract_sensitive()

**Description:** Takes a pandas dataframe and extract sensitive attributes 
from list of attributes.

**Parameters:**         
- dataframe (pandas dataframe): Dataframe to translate 
- attributes: list of sensitive attributes

>encode_dummies()

**Description:** Dummy encodes dataframe.

```
dummy = pd.get_dummies(df, prefix_sep=".", drop_first=True)
return dummy
```
</details>

## fairness
Module for fairness evaluation
<details>
<summary>Click here for documentation</summary>

>parity_score()

**Description:**
Demographic parity is defined as

$$
P(\hat{Y} | S = 0) = P(\hat{Y} | S = 1)
$$

Where $\hat{Y}$ is the predictor and $S$ is the sensitive attribute.

This can be generalised to a multiclass case with $K$ classes.

$$
P(\hat{Y} | S_i) = P(\hat{Y} | S_j) \qquad i, j \in \{0, \dots, K-1\}
$$

We want to condense this to a single metric between $0$ and $1$. I.e, when we 
have likelihood for the different classes of a sensitive attribute in a list of 
probabilities $L$ like so

$$
L =\{ P(\hat{Y} | S=0), \dots, P(\hat{Y} | S=K-1) \}
$$

and for that, we have worked out the following funsction $f$

$$
    f = \frac{\text{geometric mean}(L)}{\text{mean}(L)}
$$

**Parameters:**
  - probabilities (list): list of sensitive conditional probabilities.

```
def parity_score(probabilities):
    a = np.array(probabilities)
    return a.prod() ** (1.0 / len(a)) / a.mean()
```

>fairness_report()
**Description:**

Fairness report.

Calculates some fairness and performance metrics from test labels and predictions. Returns a dataframe of results.

**Parameters:**
  - y (array): Dataset Labels.
  - y_pred (array): Model predictions.
  - sensitives (dataframe): Test dataset of sensitive attributes.
  - model_name (string): Name of model.

</details>

## Tree
Module for tree-based models.

<details>
<summary>Click here for documentation</summary>

### class **FairDecisionTreeClassifier**

Fair Decision Tree Classifier. The code for this class is borrowed from this [GitHub repository](https://github.com/pereirabarataap/fair_tree_classifier). This decision tree evaluates candidate splits using Splitting
Criterion AUC for Fairness (SCAFF). See their [paper](https://scholar.google.com/scholar_url?url=https://www.researchgate.net/profile/Antonio-Pereira-Barata-2/publication/355391905_Fair_Tree_Classifier_using_Strong_Demographic_Parity/links/61f14fab5779d35951d60684/Fair-Tree-Classifier-using-Strong-Demographic-Parity.pdf&hl=no&sa=T&oi=gsb-ggp&ct=res&cd=0&d=11839880095099199982&ei=Iw9QYomJIszBsQKLmZzIDQ&scisig=AAGBfm2w5jFZqXIOQ5j2km5xwLFunTPXGg) for more details.


>fit()

**Description:** 

Trains the decision tree using the traditional algorithm of generating candidate
splits, evaluating split in terms of the chosen splitting criterion (SCAFF) and
selecting the best split.

**Parameters:**
- X -> any_dim pandas.df or np.array: numerical/categorical
- y -> one_dim pandas.df or np.array: only binary
- b (bias) -> any_dim pandas.df or np.array: treated as str

>predict_proba()

**Description:** 

Predict the class of of feature vectors. Predictions are calculated as 
probabilities of belonging to each class. 


**Parameters:**

- X -> any_dim pandas.df or np.array: numerical/categorical

>predict()

Predict the class of of feature vectors. Predictions are outputted as class 
the feature vector is classified to. 

**Parameters:**

- X -> any_dim pandas.df or np.array: numerical/categorical

### class **FairRandomForestClassifier**

This classifier learns several decision trees using tradition random forest 
methods. The decision trees are trained on bootstrapped dataset with removed 
columns etc.


>fit()

**Description:** 

Trains the random forestusing the traditional algorithm of bootstrapping
datasets and removing features from teh dataset.

**Parameters:**
- X -> any_dim pandas.df or np.array: numerical/categorical
- y -> one_dim pandas.df or np.array: only binary
- b (bias) -> any_dim pandas.df or np.array: treated as str

>predict_proba()

**Description:** 

Predict the class of of feature vectors. Predictions are calculated as 
probabilities of belonging to each class. 


**Parameters:**

- X -> any_dim pandas.df or np.array: numerical/categorical

>predict()

Predict the class of of feature vectors. Predictions are outputted as class 
the feature vector is classified to. 

**Parameters:**

- X -> any_dim pandas.df or np.array: numerical/categorical

</details>

## Datasets
<details>
<summary>Click here for documentation</summary>

### Generating synthetic datasets

>datasetgen_numerical()

**Description**:

Creates a dataset with two sensitive columns (Race and Gender) as well as
4 gaussian features.

**Parameters**:

- n_samples (integer): Number of datapoints in dataset.
- informative (bool): Is dataset informative? True or False.
- seperability (float): Parameter for seperating sensitive classes.


**Returns**:
- df: Pandas DataFrame.
</details>