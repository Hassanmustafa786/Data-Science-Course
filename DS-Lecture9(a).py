#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# #### Scatter Plot

# In[11]:


x = np.linspace(0,10,30)
y = np.sin(x)

plt.plot(x,y, '.', color = "black")
plt.show()


# In[12]:


plt.scatter(x,y)
plt.show()


# In[13]:


plt.plot(x,y, '-o', color = "black")
plt.show()


# In[14]:


plt.plot(x,y, '-p', color = "black") #Hexagonal Shape
plt.show()


# In[15]:


x = np.random.randn(100)
y = np.random.randn(100)
colors = np.random.rand(100)
sizes = 1000 * np.random.rand(100)

plt.scatter(x,y, c = colors, s= sizes, alpha = 0.3)
plt.colorbar()
plt.colormaps()
plt.show()


# In[16]:


data = np.random.randn(1000)
plt.hist(data)


# In[17]:


plt.hist(data, bins=20, alpha=0.5, histtype="stepfilled" , color="red", )
plt.show


# In[18]:


x1 = np.random.normal(0,0.8,1000)
x2 = np.random.normal(-2,1,1000)
x3 = np.random.normal(3,2,1000)

dict_extra = dict(histtype="stepfilled", alpha=0.5, bins=30)

plt.hist(x1, **dict_extra) #1 star = list
plt.hist(x2, **dict_extra) #2 star = dictionary that'swhy we use this 2 stars.
plt.hist(x3, **dict_extra)
plt.show


# In[26]:


x = np.linspace(0,10,1000)

plt.plot(x, np.sin(x), "-b", label= "Sin Graph")
plt.plot(x, np.cos(x), "--r", label= "Cos Graph")
plt.legend(loc="lower right", ncol=2, fancybox=True, borderpad=1, shadow=True)
plt.show()


# #### Exploratory Data Analysis (EDA)

# In[29]:


data = pd.read_csv("netflixData.csv")
data.head()


# In[31]:


data.shape


# In[34]:


data.isnull().sum()


# In[35]:


import seaborn as sns


# In[36]:


sns.heatmap(data.isnull())


# In[37]:


sns.countplot(x="Content Type", data=data)


# In[38]:


sns.countplot(x="Production Country", data=data)


# In[ ]:




