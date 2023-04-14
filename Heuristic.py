#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score


# ## Load data and add column names

# In[27]:


dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz", header=None)


# In[28]:


dataset.columns = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"] + [
    "Wilderness_Area_{}".format(i) for i in range(4)] + [
    "Soil_Type_{}".format(i) for i in range(40)] + [
    "Cover_Type"
]


# ## Exploring the data

# In[29]:


dataset.head()


# In[30]:


dataset.tail()


# In[31]:


dataset.describe()


# In[33]:


dataset.corr()


# ## Check for missing data

# In[32]:


dataset.isnull().sum()


# In[34]:


import matplotlib.pyplot as plt


# ## Creating a chart for better analysis

# In[35]:


def plot_boxplots(dataset, columns):
    for col in columns:
        fig, ax = plt.subplots(figsize=(6, 6))
        dataset.boxplot(column=col, by="Cover_Type", ax=ax)
        ax.set_title(col)
        ax.set_xlabel("Cover Type")
        ax.set_ylabel(col)
        plt.show()


# In[36]:


plot_boxplots(dataset, dataset.iloc[:, :4].columns)


# ## Implementation of heursitic logic

# In[44]:


def heuristic_model(data: pd.DataFrame) -> list:
    predictions = []
    for i in range(0, len(data)):
        value = data.iloc[i][0]

        if value >= 0 and value < 2000:
            predictions.append(3)
        
        if value >= 2000 and value < 2300:
            predictions.append(4)
        
        if value >= 2300 and value < 2600:
            predictions.append(3)
        
        if value >= 2600 and value < 2900:
            predictions.append(5)
        
        if value >= 2900 and value < 3100:
            predictions.append(2)
        
        if value >= 3100 and value < 3250:
            predictions.append(1)
        
        if value >= 3250: 
            predictions.append(7)

    return predictions 


# In[46]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ## Model evaluation

# In[47]:


X_train, X_test, y_train, y_test = train_test_split(dataset.drop(["Cover_Type"], axis = 1), dataset["Cover_Type"], test_size=0.2)
y_pred = heuristic_model(X_test)


# In[48]:


accuracy = accuracy_score(y_test, y_pred)
print("Heuristic - accuracy:", accuracy)


# In[51]:


H_f1_score = f1_score(y_test, y_pred,average='macro')
print("Heuristic - F1 score: ", H_f1_score)


# In[53]:


H_precision = precision_score(y_test, y_pred, average='macro')
print("Heuristic - Precision: ", H_precision)

