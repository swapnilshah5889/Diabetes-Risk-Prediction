# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 10:53:53 2020

@author: swapn
"""
import numpy as nm 
import matplotlib.pyplot as plt 
import pandas as pd 
import diabetes as dia      
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt


def test_neural_net(dataset,X):
    model = tf.keras.models.load_model('my_model.h5')
    
    result = model.predict(X )
    dataset['neural-net-rf'] = result


def test_neural_net2(dataset,X):
    model = tf.keras.models.load_model('neuralnet2.h5')
    
    result = model.predict(X )
    dataset['neural-net2-rf'] = result

def test_random_forest(dataset,X):
    loaded_model = pickle.load(open('randomforestmodel.sav','rb'))
    dataset['random-forest-rf'] = loaded_model.predict(X);


def plot_results(dataset):
    
    plt.plot(dataset['id'], dataset['risk-factor'], '-b',label='actual')
    plt.plot(dataset['id'], dataset['neural-net-rf'], '-r',label='neural-net')
    plt.plot(dataset['id'], dataset['neural-net2-rf'], '-y',label='neural-net-2')
    plt.plot(dataset['id'], dataset['random-forest-rf'], '-g',label='random-forest')
    plt.title('Risk Factor')
    plt.xlabel('patients')
    plt.xticks(dataset['id'])
    plt.yticks([0, 10, 20, 30, 40, 50, 60])
    plt.ylabel('risk factor')
    plt.legend(loc="upper left")
    plt.show()

if __name__ == "__main__":
    dataset = pd.read_csv('test_data.csv')
    X = dataset.iloc[:, 5:].values
    dataset['risk-factor'] =  dataset.apply(dia.calculate_risk,axis=1)
    test_neural_net(dataset,X)
    test_random_forest(dataset, X)
    X1 = dataset.iloc[:, 1:7].values
    test_neural_net2(dataset, X1)
    plot_results(dataset)