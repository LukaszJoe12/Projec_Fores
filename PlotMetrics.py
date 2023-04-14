#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[52]:


data = {
    'Metrics': ['Accuracy', 'F1 - score', 'Precision'],
    'Heuristic': [0.427, 0.31, 0.307],
    'Logistic Regression': [0.723, 0.522, 0.591],
    'Random Forest': [0.943, 0.909, 0.935],
    'Deep Learning': [0.901, 0.854, 0.871]
}

df = pd.DataFrame(data)


# In[53]:


def plot_model_metrics(df):
    models = df.columns[1:]
    metrics = df.iloc[:, 0]
    num_models = len(models)

    bar_width = 0.25
    opacity = 0.8

    for i, metric in enumerate(metrics):
        values = df.iloc[i, 1:].astype(float)
        index = np.arange(num_models) + i * bar_width

        plt.bar(index, values, bar_width,
                alpha=opacity,
                color=plt.cm.tab20(i),
                label=metric)

    plt.xlabel('Models')
    plt.ylabel('Metrics')
    plt.xticks(np.arange(num_models) + bar_width, models)
    plt.legend()

    plt.show()


# In[54]:


plot_model_metrics(df)

