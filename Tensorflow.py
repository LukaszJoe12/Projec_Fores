#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, jaccard_score


# ## Load data and add column names

# In[9]:


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

# In[10]:


X_train, X_test, y_train, y_test = train_test_split(dataset.drop(["Cover_Type"], axis=1), dataset["Cover_Type"], test_size=0.2, random_state = 2)


# ## Nomralize the data 

# In[11]:


scaler = MinMaxScaler()
X_train_prescaled = scaler.fit_transform(X_train)
X_test_prescaled = scaler.transform(X_test)


# In[12]:


y_train = y_train - 1
y_test = y_test - 1


# In[13]:


y_train_onehot = np.eye(7)[y_train]
y_test_onehot = np.eye(7)[y_test]


# In[7]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# In[8]:


from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# ## Creating a function to find the best parameters

# In[9]:


def create_model(hidden_layers=1, neurons=64, learning_rate=0.001, reg_strength=0.01):
    model = Sequential()
    model.add(Dense(neurons, input_dim = X_train_prescaled.shape[1], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))

    for i in range(hidden_layers):
        model.add(Dense(neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))

    model.add(Dense(7, activation='softmax'))
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

keras_clf = KerasClassifier(build_fn=create_model, epochs=50, batch_size=32, verbose=1)


# In[10]:


param_distribs = {
    "hidden_layers": [1, 2, 3],
    "neurons": [32, 64, 128],
    "learning_rate": [0.001, 0.01, 0.1],
    "reg_strength": [0.01, 0.001, 0.0001]
}

rnd_search_cv = RandomizedSearchCV(keras_clf, param_distribs, n_iter=5, cv=3)

rnd_search_cv.fit(X_train_prescaled, y_train_onehot)


# In[11]:


print("Best params:", rnd_search_cv.best_params_)
print("Best score:", rnd_search_cv.best_score_)


# ## Training the model with the best parameters

# In[12]:


best_model = create_model(**rnd_search_cv.best_params_)


# In[13]:


DP = best_model.fit(X_train_prescaled, y_train_onehot, epochs=100, validation_data=(X_test_prescaled, y_test_onehot))


# ## Save the trained model

# In[14]:


best_model.save("deep_learning.h5")


# ## Plot training curves for the best hyperparameters

# In[15]:


import matplotlib.pyplot as plt

plt.plot(DP.history['accuracy'])
plt.plot(DP.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(DP.history['loss'])
plt.plot(DP.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# ## Model evaluation

# In[14]:


from tensorflow import keras


# In[15]:


best_model = keras.models.load_model('deep_learning.h5')


# In[16]:


y_hat = best_model.predict(X_test_prescaled)


# In[17]:


y_pred = np.argmax(y_hat, axis=1)
y_true = np.argmax(y_test_onehot, axis=1)

accuracy = accuracy_score(y_true, y_pred)
print('Accuracy - Deep Learning:', accuracy)


# In[18]:


DP_f1_score = f1_score(y_true, y_pred, average='macro')
print("Deep Learning - F1 score: ", DP_f1_score)


# In[19]:


DP_precision = precision_score(y_true, y_pred, average='macro')
print("Deep Learning - Precision: ", DP_precision)


# In[ ]:




