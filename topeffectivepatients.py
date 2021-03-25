# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:12:56 2020

@author: hardik
"""


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import diabetes as dia      
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dense
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.keras.optimizers import SGD

dataset = pd.read_csv('diabetes_data_2.7_with_risk.csv')

#dataset['finalcall']= dataset['risk-factor'].apply(lambda x: 1 if x>140 else 0 )
m = dataset.iloc[:, 2:12].values
y = dataset.iloc[:, -1].values


   
scalar = StandardScaler().fit(m)
X= scalar.transform(m)
scalar1 = MinMaxScaler() 
y = scalar1.fit_transform(y.reshape(-1,1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


model = tf.keras.Sequential()
model.add(Dense(11, input_dim=10, activation='relu',
 use_bias=True))
#model.add(Dense(4, activation='relu', use_bias=True))
#model.add(Dense(4, activation='sigmoid', use_bias=True))
model.add(Dense(20, activation="relu",use_bias=True))
model.add(Dense(20, activation="relu",use_bias=True))
model.add(Dense(1, activation="linear"))

model.compile(loss='mean_squared_error',
 optimizer='adam',
 metrics=['binary_accuracy'])
print (model.get_weights())
history = model.fit(X_train, y_train, epochs=20, validation_data = (X_train, y_train))
model.summary()


model.save('topeffectivepatients.h5')

y_pred = model.predict(X_test )
#dataset['predictedvalue']=model.predict(X)



k=np.arange(1,38881,200)
risk_factor_p = y_pred[:,0][k]
#pred_ans = y_pred[:,1][k]
print (y_pred)
scores = model.evaluate(X_test, y_test, verbose=0)



plt.plot(k, y_test[k], '-b',label='test data')
plt.plot(k, y_pred[k], '-r',label='predicted data')
plt.title('Neural net')
plt.xlabel('patients')
plt.ylabel('risk factor')
plt.legend(loc="upper left")
plt.show()

