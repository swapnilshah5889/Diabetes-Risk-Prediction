import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import diabetes as dia      
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dense


def fbs_low(x):
    
    if x < 80 and x>=60:
        return 1
    else:
        return 0
    
def fbs_vlow(x):
    
    if x < 60 and x>=50:
        return 1
    else:
        return 0
    
def fbs_exlow(x):
    
    if x < 50:
        return 1
    else:
        return 0    

def fbs_high(x):
    
    if x >= 140 and x < 250:
        return 1
    else:
        return 0

def fbs_vhigh(x):
    
    if x >= 250 and x<=400:
        return 1
    else:
        return 0

def fbs_exhigh(x):
    
    if x > 400:
        return 1
    else:
        return 0

def creatinine_high(x):
    if x > 1.5 and x<3:
        return 1
    else:
        return 0

def creatinine_vhigh(x):
    if x > 3:
        return 1
    else:
        return 0

def hba1c_pre(x):
    if x >= 6 and x<=7:
        return 1
    else:
        return 0

def hba1c_dia(x):
    if x > 7 and x <= 10:
        return 1
    else:
        return 0


def hba1c_dia_high(x):
    if x > 10 and x <= 15 :
        return 1
    else:
        return 0
    

def hba1c_dia_vhigh(x):
    if x > 15:
        return 1
    else:
        return 0


dataset = pd.read_csv('diabetes_data_2.7.csv')

dataset=dataset[['id','fbs','pp2bs','creatinine','hba1c','fbs_normal','pp2bs_normal','days_from_last_appointment','urine_ketoacidosis','fundus_retinotherapy','age']]

dataset['fbs-exlow'] = dataset['fbs'].apply(fbs_exlow)
dataset['fbs-vlow'] = dataset['fbs'].apply(fbs_vlow)
dataset['fbs-low'] = dataset['fbs'].apply(fbs_low)
dataset['fbs-high'] = dataset['fbs'].apply(fbs_high)
dataset['fbs-vhigh'] = dataset['fbs'].apply(fbs_vhigh)
dataset['fbs-exhigh'] = dataset['fbs'].apply(fbs_exhigh)

dataset['pp2bs-exlow'] = dataset['pp2bs'].apply(fbs_exlow)
dataset['pp2bs-vlow'] = dataset['pp2bs'].apply(fbs_vlow)
dataset['pp2bs-low'] = dataset['pp2bs'].apply(fbs_low)
dataset['pp2bs-high'] = dataset['pp2bs'].apply(fbs_high)
dataset['pp2bs-vhigh'] = dataset['pp2bs'].apply(fbs_vhigh)
dataset['pp2bs-exhigh'] = dataset['pp2bs'].apply(fbs_exhigh)

dataset['creatinine-high'] = dataset['creatinine'].apply(creatinine_high)
dataset['creatinine-vhigh'] = dataset['creatinine'].apply(creatinine_vhigh)

dataset['hba1c-pre'] = dataset['hba1c'].apply(hba1c_pre)
dataset['hba1c-dia'] = dataset['hba1c'].apply(hba1c_dia)
dataset['hba1c-dia_high'] = dataset['hba1c'].apply(hba1c_dia_high)
dataset['hba1c-dia_vhigh'] = dataset['hba1c'].apply(hba1c_dia_vhigh)

#dataset = pd.read_csv('diabetes_data_low_with_risk.csv')

dataset['risk-factor'] =  dataset.apply(dia.calculate_risk1,axis=1)
dataset.set_index('id',inplace=True)
dataset.to_csv('diabetes_data_2.7_with_risk.csv',encoding='utf-8',index=True)


dataset = pd.read_csv('diabetes_data_2.7_with_risk.csv')
X = dataset.iloc[:,13:-1].values

y = dataset.iloc[:,-1].values


model = tf.keras.Sequential()
model.add(Dense(21, input_dim=20, activation='relu',
 use_bias=True))
#model.add(Dense(4, activation='relu', use_bias=True))
#model.add(Dense(4, activation='sigmoid', use_bias=True))
model.add(Dense(1, activation="linear"))

model.compile(loss='mean_squared_error',
 optimizer='adam',
 metrics=['binary_accuracy'])
print (model.get_weights())
history = model.fit(X, y, epochs=20,
 validation_data = (X, y))
model.summary()


model.save('neuralnet2.h5')

result = model.predict(X )
print (result)
scores = model.evaluate(X, y, verbose=0)


error=0
h = y.size
for i in range(h):
    diff = abs(y[i]-result[i])
    if diff>1:
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
#dataset.to_csv('diabetes_data_low.csv',encoding='utf-8',index=False)


