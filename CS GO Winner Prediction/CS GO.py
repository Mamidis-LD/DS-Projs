import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#Reading the data
data = pd.read_csv('C:\\Users\\EMAMSUR\\Documents\\DS Projects\\All DS Exercises\\CS GO Winner Prediction\\csgo_round_snapshots.csv')
data.info()
data
#Checking for any null values
np.sum(np.sum(data.isnull()))
#Filtering only integer values of dataset
data.select_dtypes(np.number)
#dropping the integer features
data.drop(data.select_dtypes(np.number),axis=1)
#feature 'bomb_planted' 
data['bomb_planted'] = data['bomb_planted'].astype(np.int16)
data.select_dtypes(np.number)
data.select_dtypes(np.number)
#Hot encoding the features
encoder = LabelEncoder()
data['map'] = encoder.fit_transform(data['map'])
map_mappings = {index: label for index, label in enumerate(encoder.classes_)}
map_mappings

data['round_winner'] = encoder.fit_transform(data['round_winner'])
winner_mappings = {index: label for index, label in enumerate(encoder.classes_)}
winner_mappings
data.describe()

#Splitting the data into dependent and independent variables
Y = data['round_winner']
X = data.drop('round_winner',axis=1)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)
X

#Applying to PCA
pca = PCA(n_components=96)
pca.fit(X)

pca.explained_variance_ratio_

plt.figure(figsize=(10,10))
plt.hist(pca.explained_variance_ratio_, bins=96)
plt.show()

def getKComponents(pca,alpha):
    total_variance = 0
    for feature, variance in enumerate(pca.explained_variance_ratio_):
        total_variance += variance
        if total_variance >= 1-alpha:
            return feature + 1
    return len(pca.explained_variance_ratio_)


k = getKComponents(pca, 0.05)
X = pca.transform(X)[:,0:k]
pd.DataFrame(X)

#Splitting the data to train and test set
X_train, X_test,y_train,y_test = train_test_split(X,Y, train_size=0.8)

#Creating model
log_model = LogisticRegression(verbose=True)
nn_model = MLPClassifier(verbose=True)

log_model.fit(X_train, y_train)
nn_model.fit(X_train, y_train)

print(f'logistic Model : {log_model.score(x_test, y_test)}')
print(f'N_N Model : {nn.model.score(x_test, y_test)}')



