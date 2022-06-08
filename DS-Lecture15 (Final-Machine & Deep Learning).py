#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[11]:


data = pd.read_csv("sonar.all-data.csv")
data.head()


# In[12]:


data.shape


# In[13]:


pd.set_option('display.max_rows', 500)
data.dtypes


# In[14]:


data.groupby('R').size()


# # MODEL EVALUATION

# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix


# In[16]:


arr = data.values

X = arr[:,0:-1]
y = arr[:,-1]
testing_size = 0.2
print(y)


# In[17]:


X = data.drop('R', axis=1)
y = data['R']
print(y)


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing_size)
print(y_test) # This is the output....


# In[19]:


DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)
acc = accuracy_score(y_test, y_pred)*100
print(f"Decision Tree accuracy score is: {acc}")


# In[20]:


models = []
models.append(('DT', DecisionTreeClassifier()))
models.append(('RT', RandomForestClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LR', LogisticRegression()))
models.append(('GNB', GaussianNB()))


# In[21]:


print(models)


# In[22]:


names = []
results = []


# In[23]:


for name, model in models:
    obj = model
    names.append(name)
    obj.fit(X_train, y_train)
    y_pred = obj.predict(X_test)
    results.append(accuracy_score(y_test, y_pred)*100)
    print(results)


# In[24]:


plt.scatter(names, results)


# 
# # DEEP LEARNING

# In[25]:


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


# In[26]:


data['R'].replace(to_replace=['R','M'], value=[0,1], inplace=True)

X = data.drop('R', axis=1)
y = data['R']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing_size)


# In[27]:


#input_dim mai hmesha number of features jayenge (Means columns)...
ann_model = Sequential([
    Dense(24, input_dim = 60, activation='relu'),
    Dense(36, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann_model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
    
ann_model.summary()


# In[28]:


ann_model.fit(X_train, y_train, epochs=20)


# In[ ]:





# In[ ]:




