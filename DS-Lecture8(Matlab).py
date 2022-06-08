#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[2]:


x = np.linspace(0,10,1000)

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

plt.show


# In[3]:


x = np.linspace(0,10,1000)

fig = plt.figure()

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

fig.savefig("LinePlot.png")

plt.show
#(C:\Users\perfect\Data Science Course) is location pr esi ek pic milegi jo is method se download hui hai. 


# In[11]:


plt.Figure()

plt.subplot(2,1,1)
plt.plot(x, np.sin(x))

plt.subplot(2,1,2)
plt.plot(x, np.cos(x))

plt.show()


# In[ ]:




