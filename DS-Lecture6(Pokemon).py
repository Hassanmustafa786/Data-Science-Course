#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


data = pd.read_csv("Pokemon.csv")
data.head(n=15)


# In[3]:


data.shape


# In[4]:


data.describe()


# In[5]:


data.isnull().sum()


# In[6]:


data.info()


# In[7]:


data["Type 1"].value_counts()


# In[13]:


#Defense wala column mai small value top pr or large value bottom pr.
data.sort_values("Defense")


# In[14]:


#Isme hm ne pichle kaam ulat krdia.
data.sort_values("Defense", ascending = False)


# In[15]:


data.sort_values(["Defense", "HP"])


# In[16]:


data.sort_values(["Defense","HP"], ascending = [False,True])


# ## Feature Engineering
# ### Add Column

# In[17]:


attack_mean = data["Attack"].mean()

def set_attack(val):
    if val < attack_mean:
        return "Attack Low"
    elif val == attack_mean:
        return "Attack Neutral"
    else:
        return "Attack High"


# In[18]:


data["Attack_high_low"] = data["Attack"].apply(set_attack)
data


# ### GROUP BY

# In[19]:


data.groupby("Type 1").groups


# In[20]:


data.head()


# In[ ]:





# In[ ]:




