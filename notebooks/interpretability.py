# %% [markdown]
# # Interpretable Machine Learning
# 
# In this notebook, we will train intepretable machine learning models and 
# visualise the decision rules that it learns.

# %%
import pandas as pd
import sys
import os
from sklearn.metrics import plot_roc_curve, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pgmpy.models import NaiveBayes

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from forseti.datproc import translate_categorical
from forseti.bayesnet import interpretableNaiveBayes, latentLabelClassifier
from forseti.tree import interpretableTree

sns.set_style('darkgrid')

# %% [markdown]
# ## Decision Tree: Adult Dataset

# %%
df = pd.read_csv('data/adult.csv')
label = 'income'
sensitives = ['gender', 'race']
test_size = 0.33

for col in df.select_dtypes('object').columns:
    df[col] = df[col].astype('category')

df[label] = df[label].cat.codes

X = df.drop(label, axis=1)
y = df[label]

categorical_features = X.select_dtypes('category').columns
Xcat = pd.get_dummies(X.select_dtypes('category'), drop_first=True)
X = X.drop(categorical_features, axis=1)
X = X.join(Xcat)

(
    X_train,
    X_test1,
    y_train,
    y_test1
 ) = train_test_split(X, y, test_size=test_size)

# %% [markdown]
# Train the model on the training data.

# %%
clf1 = DT(
    criterion = 'entropy',
    splitter = 'best',
    max_depth = 4
)

clf1.fit(X_train, y_train)

# %% [markdown]
# Plot decision tree

# %%
ranked = np.argsort(clf1.feature_importances_)[::-1]
names = X.columns[ranked]
values = clf1.feature_importances_[ranked]

importance = pd.DataFrame(
    {
        'Attribute': names,
        'Importance': values
    }
)

importance = importance.loc[~(importance['Importance']==0.0)]
print(importance.to_latex())

# %% [markdown]
# ## Decision Tree: Compas Dataset

# %%
df = pd.read_csv('data/compas-two-yrs-recidivism.csv')
label = 'two_year_recid'

features = [
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

df = df[features]
sensitives = ['sex', 'race']
test_size = 0.33

for col in df.select_dtypes('object').columns:
    df[col] = df[col].astype('category')

X = df.drop(label, axis=1)
y = df[label]

categorical_features = X.select_dtypes('category').columns
Xcat = pd.get_dummies(X.select_dtypes('category'), drop_first=True)
X = X.drop(categorical_features, axis=1)
X = X.join(Xcat)

(
    X_train,
    X_test2,
    y_train,
    y_test2
 ) = train_test_split(X, y, test_size=test_size)

# %%
clf2 = DT(
    criterion = 'entropy',
    splitter = 'best',
    max_depth = 5
)

clf2.fit(X_train, y_train)

# %%
ranked = np.argsort(clf2.feature_importances_)[::-1]
names = X.columns[ranked]
values = clf2.feature_importances_[ranked]

importance = pd.DataFrame(
    {
        'Attribute': names,
        'Importance': values
    }
)

importance = importance.loc[~(importance['Importance']==0.0)]
importance

# %% [markdown]
# ## Decision Tree: Synthetic Datasets

# %%
df = pd.read_csv('data/synthethic_informative.csv', index_col=0)
label = 'Success'

sensitives = ['Sex', 'Race']
test_size = 0.33

for col in df.select_dtypes('object').columns:
    df[col] = df[col].astype('category')

X = df.drop(label, axis=1)
y = df[label]

categorical_features = X.select_dtypes('category').columns
Xcat = pd.get_dummies(X.select_dtypes('category'))
X = X.drop(categorical_features, axis=1)
X = X.join(Xcat)

# %%
(
    X_train,
    X_test3,
    y_train,
    y_test3
 ) = train_test_split(X, y, test_size=test_size)

# %%
clf3 = DT(
    criterion = 'entropy',
    splitter = 'best',
    max_depth = 4
)

clf3.fit(X_train, y_train)

# %%
ranked = np.argsort(clf3.feature_importances_)[::-1]
names = X.columns[ranked]
values = clf3.feature_importances_[ranked]

importance = pd.DataFrame(
    {
        'Attribute': names,
        'Importance': values
    }
)

importance = importance.loc[~(importance['Importance']==0.0)]
importance

# %%
plt.figure()
a=plot_roc_curve(clf1, X_test1, y_test1, name='Adult');
b=plot_roc_curve(clf2, X_test2, y_test2, ax=a.ax_, name='Compas');
c=plot_roc_curve(clf3, X_test3, y_test3, ax=b.ax_, name='Synthetic');
plt.title('ROC Curve Decision Tree')
plt.savefig(
    'figures/dt-roc.png',
    dpi=300
)
plt.close()

# %% [markdown]
# ## Naive Bayes: Adult Dataset

# %%
df = pd.read_csv('data/adult.csv')
label = 'income'

model = interpretableNaiveBayes()
model.train(
    label,
    df,
    'NBSens'
)

df1 = model.KLDWeights()
permw1 = model.PermutationImportance(1, 'NBSens');

# %% [markdown]
# ### Naive Bayes Without Sensitive

# %%
df = pd.read_csv('data/adult.csv')
sensitives = ['gender', 'race']
label = 'income'
df = df.drop(sensitives, axis=1)

model = interpretableNaiveBayes()
model.train(
    label,
    df,
    'NB'
)

df2 = model.KLDWeights()
permw2 = model.PermutationImportance(1, 'NB');

# %% [markdown]
# ### Merge Dataframes

# %%
modeldiff = df1.append(df2, ignore_index=True)
permw = permw1.append(permw2, ignore_index=True)

# %% [markdown]
# ### Plot model difference

# %%
sns.set_style('darkgrid')

p = sns.barplot(
    x='KLD',
    y='Attribute',
    hue='Model',
    data=modeldiff
)

p.set_title('KLD on conditional probabilities - Adult Dataset')
plt.savefig('figures/KLDimportance-adult.png', dpi=300)

# %%
sns.set_style('darkgrid')

p = sns.barplot(
    x='Weight',
    y='Attribute',
    hue='Model',
    data=permw
)

p.set_title('Permutation Weights on NB - Adult Dataset')
plt.savefig('figures/permimportance-adult.png', dpi=300)

# %% [markdown]
# ## Naive Bayes: Compas Dataset

# %%
df = pd.read_csv('data/compas-two-yrs-recidivism.csv')

features = [
    'race',
    'sex',
    'priors_count',
    'juv_fel_count',
    'juv_misd_count',
    'juv_other_count',
    'c_charge_degree',
    'two_year_recid'
]

df = df[features]
label = 'two_year_recid'

model = interpretableNaiveBayes()

model.train(
    label,
    df,
    'NBSens'
)

df1 = model.KLDWeights()
permw1 = model.PermutationImportance(1, 'NBSens');

# %% [markdown]
# ### Naive Bayes Without Sensitive

# %%
df = pd.read_csv('data/compas-two-yrs-recidivism.csv')

features = [
    'priors_count',
    'juv_fel_count',
    'juv_misd_count',
    'juv_other_count',
    'c_charge_degree',
    'two_year_recid'
]

df = df[features]
label = 'two_year_recid'

model = interpretableNaiveBayes()
model.train(
    label,
    df,
    'NB'
)

df2 = model.KLDWeights()
permw2 = model.PermutationImportance(1, 'NB');

# %% [markdown]
# ### Merge Dataframes

# %%
modeldiff = df1.append(df2, ignore_index=True)
permw = permw1.append(permw2, ignore_index=True)

# %% [markdown]
# ### Plot model difference

# %%
sns.set_style('darkgrid')

p = sns.barplot(
    x='KLD',
    y='Attribute',
    hue='Model',
    data=modeldiff
)

p.set_title('KLD on conditional probabilities - Compas Dataset')
plt.savefig('figures/KLDimportance-compas.png', dpi=300)

# %%
sns.set_style('darkgrid')

p = sns.barplot(
    x='KLD',
    y='Attribute',
    hue='Model',
    data=modeldiff
)

p.set_title('Permutation Weights on NB - Compas Dataset')
plt.savefig('figures/permimportance-compas.png', dpi=300)

# %% [markdown]
# ## Naive Bayes: Synthetic

# %%
df = pd.read_csv('data/synthethic_informative.csv', index_col=0)
label = 'Success'

model = interpretableNaiveBayes()

model.train(
    label,
    df,
    'NBSens'
)

df1 = model.KLDWeights()
permw1 = model.PermutationImportance(1, 'NBSens');

# %% [markdown]
# ### Without sensitive

# %%
sensitives=['Gender', 'Race']
df = df.drop(sensitives, axis=1)

model = interpretableNaiveBayes()
model.train(
    label,
    df,
    'NB'
)

df2 = model.KLDWeights()
permw2 = model.PermutationImportance(1, 'NB');

# %% [markdown]
# ### Merge dataframes

# %%
modeldiff = df1.append(df2, ignore_index=True)
permw = permw1.append(permw2, ignore_index=True)

# %% [markdown]
# ### Plot model difference

# %%
sns.set_style('darkgrid')

p = sns.barplot(
    x='KLD',
    y='Attribute',
    hue='Model',
    data=modeldiff
)

p.set_title('KLD on conditional probabilities - Synthetic Dataset')
plt.savefig('figures/KLDimportance-synthetic.png', dpi=300)

# %% [markdown]
# ### 

# %%
sns.set_style('darkgrid')

p = sns.barplot(
    x='Weight',
    y='Attribute',
    hue='Model',
    data=permw
)

p.set_title('Weight on conditional probabilities - Synthetic Dataset')
plt.savefig('figures/permimportance-synthetic.png', dpi=300)

# %% [markdown]
# # Fair Random Forest

# %% [markdown]
# ## Adult Dataset with $\theta$ = 0.3

# %%
df = pd.read_csv('data/adult.csv')
sensitives = ['gender', 'race']
label = 'income'

clf = interpretableTree()

clf.train(
    df,
    sensitives,
    label,
    orth=0.3
)

frfcw1 = clf.PermutationImportance(1, 'FRFC03')

# %% [markdown]
# ## Adult Dataset with $\theta$ = 0.5

# %%
df = pd.read_csv('data/adult.csv')
sensitives = ['gender', 'race']
label = 'income'

clf = interpretableTree()

clf.train(
    df,
    sensitives,
    label,
    orth=0.5
)

frfcw2 = clf.PermutationImportance(1, 'FRFC05')

# %% [markdown]
# ## Adult Dataset with $\theta$ = 0.7

# %%
df = pd.read_csv('data/adult.csv')
sensitives = ['gender', 'race']
label = 'income'

clf = interpretableTree()

clf.train(
    df,
    sensitives,
    label,
    orth=0.7
)

frfcw3 = clf.PermutationImportance(1, 'FRFC07')

# %% [markdown]
# ## Join Dataframes

# %%
frfcw = frfcw1.append(
    frfcw2.append(
        frfcw3, ignore_index=True
    ),
    ignore_index=True
)

# %%
sns.set_style('darkgrid')

p = sns.barplot(
    x='Weight',
    y='Attribute',
    hue='Model',
    data=frfcw
)

p.set_title('FRFC Permutation Importance - Adult Dataset')
plt.savefig('figures/frfc-permimportance-adult.png', dpi=300)

# %% [markdown]
# # Latent Fair Label Bayesian Network

# %%
df = pd.read_csv('data/adult.csv')
sensitives = ['gender', 'race']
label = 'income'

clf = latentLabelClassifier(
    df,
    sensitives,
    label,
    atol=0.01
)

clf.load('trained-models/fair_model.sav')

fairbnw = clf.PermutationImportance(1, name='fairBN')

# %%
sns.set_style('darkgrid')

p = sns.barplot(
    x='Weight',
    y='Attribute',
    hue='Model',
    data=fairbnw
)

p.set_title('FairBN Permutation Importance - Adult Dataset')
plt.savefig('figures/fairbn-permimportance-adult.png', dpi=300)


