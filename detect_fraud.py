import pandas as pd
import seaborn as sns
from sklearn.preprocessing import  RobustScaler # less prone to outliers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
cc_data = pd.read_csv('dataset/creditcard.csv',header=0)
cc_data.Class = pd.Categorical(cc_data.Class)
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
print(features['Amount'].describe())
# sns.distplot(cc_data.Amount.values, color='b') # data isnt in bell curve hence, will be using normalization techniques
# sns.scatterplot(x=features['Amount'],y=label, data=cc_data, hue='Class')
# plt.show()

# scale the amount and drop previous column
cc_data['scaled_amount'] = RobustScaler().fit_transform(cc_data.Amount.values.reshape(-1, 1))
cc_data.drop(['Amount'],axis=1,inplace=True)
print(cc_data.head())
X = cc_data.drop(['Class'], axis=1)

X_train, y_train, x_test, y_test = train_test_split(X, label)
clf = LogisticRegression()
clf.fit(X_train,y_train)

