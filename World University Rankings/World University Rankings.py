#!/usr/bin/env python
# coding: utf-8

# In[84]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[85]:


data = pd.read_csv('cwurData.csv')
data


# In[7]:


data.describe()


# In[8]:


data.info()


# In[16]:





# In[57]:


data.describe()


# In[45]:


sns.distplot(data['score'],bins=10, kde=True)


# In[59]:


data.drop('institution',axis=1,inplace=True)


# In[62]:


data.isnull().sum()*100/len(data)


# In[74]:


data.select_dtypes(include=np.number)


# In[93]:


df.nunique()


# In[95]:


plt.figure(figsize=(15,6))
sns.heatmap(df.corr(), annot=True)


# In[100]:


plt.figure(figsize=(15,6))
sns.barplot(df['world_rank'].head(100),y=df['publications'])


# In[118]:


data.drop('broad_impact',axis=1,inplace=True)


# In[119]:


data


# In[120]:


X=data.drop('world_rank',axis=1)
y=data['world_rank']


# In[123]:


scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)


# In[125]:


X


# In[126]:


X_train, X_test, y_train, y_test= train_test_split(X,y , train_size=0.8,random_state=10)


# In[128]:


model = LinearRegression()
model.fit(X_train,y_train)


# In[133]:


from sklearn.svm import SVR
model = SVR()
model.fit(X_train,y_train)


# In[134]:


print(f"Model R^2: {model.score(X_test, y_test)}")


# In[ ]:




