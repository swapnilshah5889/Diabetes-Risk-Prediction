import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import diabetes as dia      
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dense



dataset = pd.read_csv('diabetes_data_2.7_with_risk.csv')


X = dataset.iloc[:, 5:-1].values
y = dataset.iloc[:, -1].values





model = tf.keras.Sequential()
model.add(Dense(12, input_dim=11, activation='relu',
 use_bias=True))
#model.add(Dense(4, activation='relu', use_bias=True))
#model.add(Dense(4, activation='sigmoid', use_bias=True))
model.add(Dense(6, activation="relu"))
model.add(Dense(1, activation="linear"))


model.compile(loss='mean_squared_error',
 optimizer='adam',
 metrics=['binary_accuracy'])
print (model.get_weights())
history = model.fit(X, y, epochs=20,
 validation_data = (X, y))
model.summary()


model.save('my_model.h5')

# printing out to file
loss_history = history.history["loss"]
numpy_loss_history = np.array(loss_history)
np.savetxt("loss_history.txt", numpy_loss_history,
 delimiter="\n")
binary_accuracy_history = history.history["binary_accuracy"]
numpy_binary_accuracy = np.array(binary_accuracy_history)
np.savetxt("binary-accuracy.txt", numpy_binary_accuracy, delimiter="\n")
print(np.mean(history.history["binary_accuracy"]))
result = model.predict(X )
print (result)
scores = model.evaluate(X, y, verbose=0)