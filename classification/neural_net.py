# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:49:43 2020

@author: swapn
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd   
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import seaborn as sns

dataset = pd.read_csv('critical_data.csv')
X= dataset.iloc[:,2:-1]
y= dataset.iloc[:,-1]

sc = StandardScaler().fit(X)
X = sc.transform(X)

#dataset.describe()



def refit_model():
    from keras.models import load_model
    model = load_model('neural_net.h5')
    df = pd.read_csv('test_data.csv')
    test_input = df.iloc[:,2:-1]
    
    test_input = sc.transform(test_input)
    #test_pred = model.predict(test_input)
    
    test_outout = df.iloc[:,-1]
    model.fit(test_input,test_outout, batch_size=10, epochs=10)
    
    test_pred = model.predict(test_input)
    
    test_pred =(test_pred>0.5)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_outout, test_pred)
    model_accuracy = (cm[0][0] + cm[1][1])/(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])*100
    model.save('neural_net.h5')


def test_model():
    from keras.models import load_model
    model = load_model('neural_net.h5')
    df = pd.read_csv('critical_data.csv')
    test_input = df.iloc[:,2:-1]
    
    test_input = sc.transform(test_input)
    test_pred = model.predict(test_input)
    
    test_pred =(test_pred>0.5)
    
    test_outout = df.iloc[:,-1]
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_outout, test_pred)
    
    model_accuracy = (cm[0][0] + cm[1][1])/(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])*100
   
    
    
    """
    incorrect_positions = []
    for pos in range(len(test_outout)):
        #print(pos)
        if test_outout[pos] == 0:
            if(test_outout[pos] != test_pred[pos]):
                incorrect_positions.append(pos)
            
    newdf = df.iloc[incorrect_positions]
    newdf['critical'] = 1
    
    new_input = newdf.iloc[:,2:-1]
    new_output = newdf.iloc[:,-1]
    
    new_input = sc.transform(new_input)
    model.fit(new_input,new_output,batch_size=10,epochs=10)
    model.save('neural_net.h5')
    """
    pass


def build_model():
    
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    from keras import Sequential
    from keras.layers import Dense
    
    
    classifier = Sequential()
    #First Hidden Layer
    classifier.add(Dense(10, activation='relu', kernel_initializer='random_normal', input_dim=10))
    #Second  Hidden Layer
    classifier.add(Dense(5, activation='relu', kernel_initializer='random_normal'))
    #Output Layer
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    
    
    #Compiling the neural network
    classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
    
    #Fitting the data to the training dataset
    classifier.fit(X_train,y_train, batch_size=10, epochs=10)
    
    classifier.save('neural_net.h5')
    
    from keras.models import load_model
    model = load_model('neural_net.h5')
    df = pd.read_csv('test_data.csv')
    test_input = df.iloc[:,2:-1]
    
    test_input = sc.transform(test_input)
    #test_pred = model.predict(test_input)
    
    test_outout = df.iloc[:,-1]
    model.fit(test_input,test_outout, batch_size=10, epochs=10)
    
    model.save('neural_net.h5')
    
    
    from keras.models import load_model
    model = load_model('neural_net.h5')
    df = pd.read_csv('dummy_normal_data.csv')
    test_input = df.iloc[:,2:-1]
    
    test_input = sc.transform(test_input)
    #test_pred = model.predict(test_input)
    
    test_outout = df.iloc[:,-1]
    model.fit(test_input,test_outout, batch_size=10, epochs=10)
    
    model.save('neural_net.h5')
    
    classifier = load_model('neural_net.h5')
    
    
    eval_model=classifier.evaluate(X_train, y_train)
    eval_model
    
    y_pred=classifier.predict(X_test)
    y_pred =(y_pred>0.5)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    model_accuracy = (cm[0][0] + cm[1][1])/(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])*100
    
    
    classifier.save('neural_net.h5')
    
    #sns.pairplot(dataset, hue='critical')
    
    sns.heatmap(dataset.corr(), annot=True)

build_model()