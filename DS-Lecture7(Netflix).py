#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("netflix_titles.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


# data.loc[<row labels>,<column labels>]
data.loc[:,'type'].value_counts()


# In[8]:


data.loc[:,["title", "release_year"]]


# In[9]:


data.loc[1,"cast"].split(", ")


# In[10]:


len(data.loc[1,"cast"].split(", "))


# In[11]:


data[["cast","director","country"]] = data[["cast","director","country"]].replace(np.nan, "None")
data


# In[12]:


data[data.release_year >= 2020].count()


# In[13]:


data.groupby("type")["rating"].value_counts()
# type mai pehle group bna diya Movie or TV-show ka phr usme jitni bhi ratings thi unka sum  nikal diya.


# In[14]:


data.groupby("type").groups


# In[15]:


data.loc[data["title"]== "Kota Factory"]
# loc mai hm condition ki basis pr bhi selection krwa skte.
# iloc mai hm indexes ki basis pr selection krwate hai.


# In[16]:


data.loc[data["title"]!= "Kota Factory"]


# #### ASSIGNMENT 3

# In[17]:


def set_cast(val):
    if val == "None":
        return 0
    else:
        return len(str(val).split(", "))


# In[18]:


data["cast"].isnull().sum()


# In[19]:


data["num_of_cast"] = data["cast"].apply(set_cast)
data


# In[22]:


data.groupby(["type","country"]).count()


# In[ ]:





# In[ ]:





# In[ ]:




