
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('C:\\Users\\EMAMSUR\\Documents\\DS Projects\\All DS Exercises\\Diamonds\\diamonds.csv').drop('Unnamed: 0',axis=1)

#data.drop(data.columns[0],axis=1, inplace = True)

y = data['price']
x = data.drop('price',axis=1)

print(f"Cuts : {len(x['cut'].unique())}")
print(x['cut'].unique())
print(x['cut'].value_counts().unique())

print(f"color : {len(x['color'].unique())}")
print(x['color'].unique())
print(x['color'].value_counts().unique())

print(f"clarity : {len(x['clarity'].unique())}")
print(x['clarity'].unique())
print(x['clarity'].value_counts().unique())

encoder = LabelEncoder()
x['cut'] = encoder.fit_transform(x['cut'])
cut_mappings = {index : label for index, label in enumerate(encoder.classes_)}
print(cut_mappings)

x['color'] = encoder.fit_transform(x['color'])
color_mappings = {index : label for index, label in enumerate(encoder.classes_)}
print(color_mappings)

x['clarity'] = encoder.fit_transform(x['clarity'])
clarity_mappings = {index : label for index, label in enumerate(encoder.classes_)}
print(clarity_mappings)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

X_train,X_test,y_train,y_test = train_test_split(x,y,train_size = 0.8)

std_model = LinearRegression()
model1 = Lasso(alpha=1)
model2 = Ridge(alpha=1)

std_model.fit(X_train, y_train)
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

print(f'-----without regularization-----:{std_model.score(X_test,y_test)}')
print(f'-----without regularization-----:{model1.score(X_test,y_test)}')
print(f'-----without regularization-----:{model2.score(X_test,y_test)}')

model2 = Ridge(alpha=0.8)
model2.fit(X_train, y_train)
print(f'-----without regularization-----:{model2.score(X_test,y_test)}')
