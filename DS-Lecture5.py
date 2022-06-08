#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


data=pd.read_csv("Netflix.csv")


# In[6]:


data.head()


# In[7]:


data.tail()


# In[9]:


data.describe()


# In[20]:


data.isnull().sum()


# In[10]:


data.shape


# In[11]:


data.columns


# In[13]:


type(data)


# # loc (Labelled based selection) & iloc (Index based selection)

# In[15]:


data.loc[(data.rating=="PG") & (data["release year"]==2016)]


# In[19]:


data.loc[(data.rating=="PG") | (data["release year"]==2011)]


# In[30]:


data.loc[(data.rating=="TV-14")&(data["user rating size"] >= 81)]


# In[33]:


data.loc[(data.rating=="PG"),['title','release year']]


# In[36]:


data.iloc[[2,5]]


# In[42]:


data.iloc[2:7,0:3]


# In[47]:


data.iloc[[2,3,5,7,9],0::4]


# In[49]:


int(data["user rating size"].mean())


# In[53]:


data.replace(to_replace=np.nan, value=80, inplace=True)


# In[54]:


data


# In[55]:


data.isnull().sum()


# In[ ]:




