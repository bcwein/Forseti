# Forseti
Fair machine learning python package.

This python package implements machine learning methods and functionality for evaluating machine learning models in terms of fairness.  See sections below for each module and classes.

# Modules

## bayesnet

Module for running bayesian networks 
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


## datproc

Module for data preprocessing.

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