#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets


# In[29]:


dt = datasets.load_iris()
print(dt)


# In[30]:


dt.feature_names


# In[31]:


df = pd.DataFrame(data= dt.data, columns= dt.feature_names)
df.head()


# In[32]:


print(dt.target)
print(dt.target_names)


# In[33]:


# 0 = setosa, 1 = versicolor, 2 = virginica
df['LABEL'] = pd.Series(dt.target)
df.head()


# In[34]:


print(df.info())
print(df.describe())


# In[35]:


df.LABEL.value_counts()


# In[36]:


plt.figure(figsize=(25,10))
sns.set(font_scale=2)
sns.countplot(data=df)


# # MACHINE LEARNING

# In[37]:


# X -> Independent Data (Train) -> features hote hai jispr aap data train krwa rhe ho
# Y -> Dependent Data (Test)


# In[41]:


X = df.drop('LABEL', axis=1)
y = df['LABEL']


# In[42]:


print(X.shape)
print(y.shape)


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# In[48]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[60]:


k_range = list(range(1,20))
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
plt.plot(k_range, scores)
plt.xlabel("Value of k for KNN")
plt.ylabel("Accuracy Scores")
sns.set(font_scale=0)
plt.show()


# In[61]:


knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)*100
print(f"Accuracy of our KNN model is : {acc}")


# In[ ]:





# In[ ]:




