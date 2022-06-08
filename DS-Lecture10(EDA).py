#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("heart.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


df.info()


# In[37]:


sns.set(font_scale=2)
df.hist(figsize=(30,30))

#fig = plt.figure()
#fig.savefig("histPlot.png")

plt.show


# In[7]:


df["target"].value_counts()


# In[32]:


sns.set(font_scale=1)
sns.countplot(x = "target", data = df)
plt.ylabel("target")


# In[9]:


df["sex"].value_counts()


# In[33]:


sns.set(font_scale=1)
sns.countplot(x = "sex", data = df)
plt.ylabel("target")


# In[34]:


sns.set(font_scale=1)
sort = df.groupby("sex")["target"].value_counts()
print(sort)
sns.countplot(x = "sex" ,hue = "target", data = df)
plt.ylabel("target")


# In[38]:


plt.figure(figsize=(25,10))
sort = df.groupby("age")["target"].value_counts()
sns.countplot(x = "age" ,hue = "target", data = df,)

#fig = plt.figure()
#fig.savefig("histPlot.png")
#plt.show


# In[13]:


a = ["eat","sleep","repeat"]
ans = enumerate(a)
print(list(ans))


# In[14]:


b = "Mufaddal"
ans1 = enumerate(b)
print(list(ans1))


# In[16]:


categorical_values = list(df.columns)


# In[17]:


categorical_values.remove("target")


# In[19]:


categorical_values


# In[41]:


plt.figure(figsize=(30,30))

for i, col in enumerate(categorical_values,1):
    plt.subplot(4,4,i)
    sns.barplot(x=f"{col}", y="target", data = df)
    plt.ylabel("Possibility of having Heart Disease")
    plt.xlabel(f"{col}")


# In[ ]:




