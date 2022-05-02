# %% [markdown]
# # Fair Tree Classifier
# 
# In this notebook, we want to train a fair decision tree implemented by 
# Barata, A. P., Takes, F. W., van den Herik, H. J., & Veenman, C. J.
# 
# in their papar *Fair Tree Classifier using Strong Demographic Parity.*
# 
# They have provided code for their classifier which is added to Forseti.

# %%
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from forseti.tree import FairRandomForestClassifier
from forseti.datproc import translate_categorical, extract_sensitive
from forseti.fairness import parity_score, fairness_report, sns_auc_score

tmp = pd.read_csv('data/adult.csv')
data, codes = translate_categorical(tmp.copy(deep=True))
sensitive_attributes = ['race', 'gender']
sensitive, features = extract_sensitive(data, sensitive_attributes)
label = 'income'

y = features[label]
s = sensitive
X = tmp[features.columns].drop(label, axis=1)

# One-Hot encode the sensitive attributes
encoder = OneHotEncoder(handle_unknown='ignore')
encode_df = pd.DataFrame(encoder.fit_transform(s[['race']]).toarray()).astype('int')

names = {}
for col in encode_df.columns:
    names[col] = 'Race_' + codes['race'][col]

encode_df = encode_df.rename(names, axis=1)

# Merge the sensitive dataframe
s = s.join(encode_df).drop('race', axis=1)

# %% [markdown]
# ## Adult Dataset: Train Classifier and Experiments.

# %%
X_train, X_test = X[:30000], X[30000:]
y_train, y_test = y[:30000], y[30000:]
s_train, s_test = s[:30000], s[30000:]

orthogonalities = [
    0.3,
    0.5,
    0.7
]

for orth in orthogonalities:
    clf = FairRandomForestClassifier(
        max_depth=5,
        orthogonality=orth
    )

    clf.fit(X_train, y_train, s_train)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred_prob = pd.DataFrame(y_pred_prob)
    y_pred = pd.DataFrame(clf.predict(X_test))
    y_pred.index = X_test.index 

    y_pred_prob.to_csv('results/pred_adult_prob_FRFC_' + str(orth) + '.csv')
    y_pred.to_csv('results/pred_adult_FRFC_' + str(orth) + '.csv')
    pickle.dump(clf, open('trained-models/Adult_FRFC_' + str(orth) +'.sav', "wb"))

# %% [markdown]
# ## Compas Dataset

# %%
scores = pd.read_csv('data/compas-two-yrs-recidivism.csv')
attr = [
    'sex',
    'age',
    'race',
    'priors_count',
    'juv_fel_count',
    'juv_misd_count',
    'juv_other_count',
    'c_charge_degree',
    'two_year_recid'
]


tmp = scores[attr]
tmp['two_year_recid'] = tmp['two_year_recid'].astype('category')

data, codes = translate_categorical(tmp.copy(deep=True))
sensitive_attributes = ['sex', 'race']
sensitive, features = extract_sensitive(data, sensitive_attributes)
label = 'two_year_recid'

y = features[label]
s = sensitive
X = tmp[features.columns].drop(label, axis=1)

# One-Hot encode the sensitive attributes
encoder = OneHotEncoder(handle_unknown='ignore')
encode_df = pd.DataFrame(encoder.fit_transform(s[['race']]).toarray()).astype('int')

names = {}
for col in encode_df.columns:
    names[col] = 'Race_' + codes['race'][col]

encode_df = encode_df.rename(names, axis=1)

# Merge the sensitive dataframe
s = s.join(encode_df).drop('race', axis=1)

# %% [markdown]
# ## Compas Dataset: Train Classifier and Experiments.

# %%
X_train, X_test = X[:6000], X[6000:]
y_train, y_test = y[:6000], y[6000:]
s_train, s_test = s[:6000], s[6000:]

orthogonalities = [
    0.3,
    0.5,
    0.7
]

for orth in orthogonalities:
    clf = FairRandomForestClassifier(
        max_depth=5,
        orthogonality=orth
    )

    clf.fit(X_train, y_train, s_train)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred_prob = pd.DataFrame(y_pred_prob)
    y_pred = pd.DataFrame(clf.predict(X_test))
    y_pred.index = X_test.index
    y_pred_prob.index = X_test.index

    y_pred_prob.to_csv('results/pred_compas_prob_FRFC_' + str(orth) + '.csv')
    y_pred.to_csv('results/pred_compas_FRFC_' + str(orth) + '.csv')
    pickle.dump(clf, open('trained-models/Compas_FRFC_' + str(orth) +'.sav', "wb"))


