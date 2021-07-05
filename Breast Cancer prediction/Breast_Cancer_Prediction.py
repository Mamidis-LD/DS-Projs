###https://www.kaggle.com/aakashverma8900/breast-cancer-eda-predictive-modelling
pip install sklearn
import missingno
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn import metrics

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
df = pd.read_csv(url, names = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst', 'Unnamed: 32'])
df.head()
df.drop(['id','Unnamed: 32'],axis=1 , inplace=True)

missingno.matrix(df)
plt.show()

df.duplicated().sum()
df.describe().T

plt.figure(figsize=(7,7))
sns.countplot(x=df['diagnosis'], palette='RdBu')
B, M = df['diagnosis'].value_counts()
print('Number of cells labelled Benign:',B)
print('Number of cells labelled Malignant:',M)
print(" ")
print('% of cells labelled benign {:.2f}'.format(round(B/len(df)*100)))
print('% of cells labelled Malign {:.2f}'.format(round(M/len(df)*100)))

#Heatmap 
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(df.corr(), annot =True, linewidth = .5,fmt='.1f',ax=ax)
plt.show()

sns.clustermap(df.corr())
plt.show()

#Checking the relationship between the specific data for understanding the relationship

sns.jointplot(x=df.loc[:,'concavity_worst'], y=df.loc[:,'concave points_worst'], kind="reg", color="#ce1414")
plt.show()

df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})

X = df.drop(["diagnosis"], axis = 1)
y = df.diagnosis.values
import numpy as np
X = (X-np.min(X))/(np.max(X)-np.min(X)).values
X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
logistic = LogisticRegression()
logistic.fit(X_train,y_train)
y_pred = logistic.predict(X_test)
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
conm = confusion_matrix(y_test,y_pred)
sns.heatmap(conm,annot=True,fmt="d")
plt.show()

