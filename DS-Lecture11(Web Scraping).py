#!/usr/bin/env python
# coding: utf-8

# In[5]:


from bs4 import BeautifulSoup
import requests
import pandas as pd


# In[2]:


page = requests.get(url="https://en.wikipedia.org/wiki/Web_scraping")
page.status_code


# In[3]:


soup = BeautifulSoup(page.content, "html.parser")

title = soup.find("h1", id="firstHeading")
print(title.text)


# In[4]:


data = soup.find("div", id="mw-content-text").text
print(data)


# In[8]:


data = pd.read_excel("data1.xlsx")


# In[16]:


print(data['Names'])
print(type(data['Names']))


# In[17]:


# Replacing the gap with underscore in column
data2 = data["Names"].str.replace(" ","_")
data2


# In[20]:


names = list(data2)
print(names)


# In[37]:


titles = []
data3 = []
for i in names:
    page = requests.get(f"https://en.wikipedia.org/wiki/{i}")
    soup = BeautifulSoup(page.content, 'html.parser')
    
    title = soup.find("h1", id="firstHeading").text
    content = soup.find("div", id="mw-content-text").text
    titles.append(title)
    data3.append(content)
    


# In[38]:


print(data3)


# In[39]:


all_data={"Names":titles, "Content":data3}


# In[40]:


all_data


# In[36]:


df = pd.DataFrame(all_data)
df.to_csv("scrapped_data.csv")


# In[ ]:





# In[ ]:





# In[ ]:




