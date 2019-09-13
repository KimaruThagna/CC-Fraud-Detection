import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression

cc_data = pd.read_csv('dataset/creditcard.csv',header=0)

print(cc_data.head())
# EDA
print("Non Fradulent Transactions:",len(cc_data.loc[cc_data['Class']==0]))
print("Fradulent Transactions:",len(cc_data.loc[cc_data['Class']==1]))
features = ['Amount'] + ['V%d' % number for number in range(1, 29)]
target = 'Class'
features = cc_data[features]
label = cc_data[target]
# plot data to examine distribution.
# If distribution is gaussian, we use standardization technique. If not, we use normalization technique
#STANDARDIZATION Assumes data is in the form of a bell curve(gaussian)
#NORMALIZATION Assumes data is not a bell curve or when you dont know the distribution of data.
# Useful when data has varying scales