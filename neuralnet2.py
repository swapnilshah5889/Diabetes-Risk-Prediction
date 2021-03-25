# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:06:47 2020

@author: swapn
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dense
from sklearn.preprocessing import StandardScaler



dataset = pd.read_csv('diabetes_data_2.7_with_risk.csv')

X = dataset.iloc[:, 7:-1].values
y = dataset.iloc[:, -1].values

model = tf.keras.Sequential()
model.add(Dense(23, input_dim=22, activation='relu',
 use_bias=True))
#model.add(Dense(4, activation='relu', use_bias=True))
#model.add(Dense(4, activation='sigmoid', use_bias=True))
model.add(Dense(12, activation="relu"))
model.add(Dense(1, activation="linear"))

model.compile(loss='mean_squared_error',
 optimizer='adam',
 metrics=['binary_accuracy'])

scalar = StandardScaler().fit(X)
rescaledX= scalar.transform(X) 
print (model.get_weights())
history = model.fit(rescaledX, y, epochs=20,
 validation_data = (rescaledX, y))
model.summary()


model.save('neuralnet2.h5')

result = model.predict(rescaledX)
print (result)
scores = model.evaluate(rescaledX, y, verbose=0)


error=0
h = y.size
for i in range(h):
    diff = abs(y[i]-result[i])
    if diff>2:
        error+=1



acc = ((h - error)/h)*100




k=np.arange(1,27648,500)
#plt.scatter(k, y_test, color = 'red')
y_new = y[k]

result_new = result[k]

plt.plot(k, y_new, '-b',label='test data')
plt.plot(k, result_new, '-r',label='predicted data')
plt.title('Neural Network')
plt.xlabel('patients')
plt.ylabel('risk factor')
plt.legend(loc="upper left")
plt.show() 
