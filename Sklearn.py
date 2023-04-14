#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler


# ## Load data and add column names

# In[63]:


dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz", header=None)
dataset.columns = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"] + [
    "Wilderness_Area_{}".format(i) for i in range(4)] + [
    "Soil_Type_{}".format(i) for i in range(40)] + [
    "Cover_Type"
]


# ## Spliting the data into training and test data

# In[64]:


X_train, X_test, y_train, y_test = train_test_split(dataset.drop(["Cover_Type"], axis=1), dataset["Cover_Type"], test_size=0.5, random_state = 2)


# In[65]:


dataset.drop(["Cover_Type"], axis=1).shape


# ## Training RandomForestClassifier

# In[66]:


RF = RandomForestClassifier(n_estimators=100)
RF.fit(X_train, y_train)


# ## Model evaluation

# In[67]:


RF_pred = RF.predict(X_test)


# In[68]:


RF_acc = accuracy_score(y_test, RF_pred)


# In[69]:


print("Random Forest - accuracy:", RF_acc)


# In[74]:


RF_f1_score = f1_score(y_test, RF_pred,average='macro')
print("Random Forest - F1 score: ", RF_f1_score)


# In[73]:


RF_precision = precision_score(y_test, RF_pred, average='macro')
print("Random Forest - Precision: ", RF_precision)


# ## Save the trained model

# In[43]:


from joblib import dump


# In[44]:


model_0 = RF
dump(model_0,'RandomForest.joblib')


# ## Data normalization

# In[45]:


scaler = MinMaxScaler()
X_train_prescaled = scaler.fit_transform(X_train)
X_test_prescaled = scaler.fit_transform(X_test)


# ## Training LogisticRegression

# In[46]:


LR = LogisticRegression(max_iter = 1000,  solver='sag')
LR.fit(X_train_prescaled, y_train)


# In[47]:


LR_predict = LR.predict(X_test_prescaled)


# In[48]:


LR_acc = accuracy_score(y_test, LR_predict)


# ## Model evaluation

# In[75]:


print("Logistic Regression - accuracy:", LR_acc)


# In[76]:


LR_f1_score = f1_score(y_test, LR_predict,average='macro')
print("Logistic Regression - F1 score: ", LR_f1_score)


# In[77]:


LR_precision = precision_score(y_test, LR_predict, average='macro')
print("Logistic Regression - Precision: ", LR_precision)


# ## Save the trained model to pickle

# In[50]:


model_1 = LR
dump(model_1, 'LogisticRegression.joblib')

