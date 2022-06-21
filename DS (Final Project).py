#!/usr/bin/env python
# coding: utf-8

# # Installing the Python Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # Loading the dataset in Notebook

# In[2]:


data = pd.read_csv("heart.csv")


# In[3]:


df = pd.DataFrame(data)
df


# # Exploratory Data Analysis

# In[4]:


df.groupby('sex').size()


# In[5]:


df.target.value_counts()


# In[6]:


sns.set(font_scale=2)
df.hist(figsize = (30,30))
plt.show


# In[7]:


sns.set(font_scale=1)

x = df['age']
y = df['thalach']
colors = np.random.randn(1025)
sizes = 1000 * np.random.randn(1025)

plt.scatter(x,y, c = colors, s= sizes, alpha = 0.3)
plt.colorbar()
plt.colormaps()
plt.title('Analysing the Age & Thalach through Graph')
plt.xlabel("Age")
plt.ylabel("Thalach")
plt.show()


# In[8]:


sns.set(font_scale=1)

plt.title('Checking the Null Values')
sns.heatmap(df.isnull())


# In[9]:


sns.set(font_scale=1)
plt.figure(figsize=(10,5))

grouped = df.groupby("sex")["target"].value_counts()
print(grouped)

sns.countplot(x= "sex", hue= "target" , data=df, palette="Set1")
plt.title("Graph Between Sex & Target for Male & Female")
plt.ylabel("target")


# In[10]:


plt.hist(x= df.age[df.target==1], bins=20, alpha=0.5, histtype="bar" , color="red", )
plt.title("Analysing the age for only male")
plt.xlabel('Age')
plt.show


# In[11]:


plt.scatter(x=df.age[df.target==1], y=df.chol[(df.target==1)], c="black")
plt.scatter(x=df.age[df.target==0], y=df.chol[(df.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Cholestrol")
plt.show()


# In[12]:


Male = len(df[df.sex==1])
Female = len(df[df.sex==0])
print("Males:", Male)
print("Females:", Female)


# # Machine Learning

# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics


# In[14]:


arr = data.values

X = arr[:,0:-1]
y = arr[:,-1]
testing_size = 0.3
print(y)


# In[15]:


X = data.drop(['target'], axis=1)
y = data['target']
print(y)


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing_size)
print(y_test) # This is the output....


# In[17]:


models = []
models.append(('DT', DecisionTreeClassifier()))
models.append(('RT', RandomForestClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LR', LogisticRegression()))
models.append(('GNB', GaussianNB()))


# In[18]:


print(models)


# In[19]:


names = []
results = []


# In[20]:


for name, model in models:
    obj = model
    names.append(name)
    obj.fit(X_train, y_train)
    y_pred = obj.predict(X_test)
    results.append(accuracy_score(y_test, y_pred)*100)
    print(results)


# In[21]:


plt.plot(names, results, '-o', color= 'black')
plt.title("Graph of Models Accuracy")
plt.xlabel("Models")
plt.show


# In[23]:


#Model Accuracy: How often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

#Model Precision: What % of +ve tuples are labelled as such?
print("Precision:", metrics.precision_score(y_test, y_pred))

#Model Recall: What % of +ve tuples are labelled as such?
print("Recall:", metrics.recall_score(y_test, y_pred))

print("Confusion Matrix:", metrics.confusion_matrix(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




