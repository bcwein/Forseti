# %% Import
import pandas as pd
import sys
import os

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from forseti.datproc import translate_categorical

# %%
df = pd.read_csv('data/adult.csv')
tmp, codes = translate_categorical(df)
# %%
codes