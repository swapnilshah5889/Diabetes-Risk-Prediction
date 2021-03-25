# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 19:19:29 2020

@author: swapn
"""
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler


def calculate_criticality(row):
    
    counter = 0
    fbs_diff = abs(row['fbs']-row['fbs_normal'])
    if fbs_diff>40:
        if row['fbs']<80 or row['fbs']>140:
            counter+=1
    
    pp2bs_diff = abs(row['pp2bs']-row['pp2bs_normal'])    
    if pp2bs_diff>40:
        if row['pp2bs'] < 80 or row['pp2bs']>140:
            counter +=1
    
    if row['hba1c'] >= 7:
        counter += 1
        
    if row['fundus_retinotherapy'] == 1:
        counter += 1

    if row['creatinine'] >= 1.5:
        counter += 1
    
    if row['urine_ketoacidosis'] == 1:
       counter += 1
    
    if counter > 0:
        print(str(row['id'])+" -> 1")
        return 1
    else:
        print(str(row['id'])+" -> 0")
        return 0
   
       
def predict_data(classifier,scalar):
    
    classifier2 = pickle.load(open('SGDClassifiermodel.sav','rb'))
    df = pd.read_csv('test_data.csv')
    
    m = df.iloc[:,2:-1]
    
    X1 = scalar.transform(m)
    
    scalar2 = MinMaxScaler().fit(m)
    
    X2 = scalar2.transform(m)
    
    
    Y1 = classifier.predict(X1)
    
    Y2 = classifier2.predict(X2)
    
    count = 0
    equality = 0
    for i in Y1:
        print(str(i)+" | "+str(Y2[count]))
        if int(i) == int(Y2[count]):
            equality += 1
        count +=1
    acc = (equality/count)*100
    print("Equality : "+str(acc))
    
if __name__=='__main__':
    
    """
    try:
        f= open("doc_ans.txt")
        print("File Opened")
    except:
        print("File Open Failed")
       
    for line in f:
        line.rstrip()
        docs_ans = line.split(',')
      
    f.close()
    """
    
    
    dataset = pd.read_csv('critical_data.csv')
    #dataset = dataset.drop('risk-factor', 1)
    
    #dataset['critical'] = dataset.apply(calculate_criticality,axis=1)
    
    
    m = dataset.iloc[:,2:-1].values
    y = dataset.iloc[:, -1].values
        
    scalar = MinMaxScaler().fit(m)
    X= scalar.transform(m) 
    #print(X)
    #print(y)


    #dataset.to_csv('critical_data.csv',encoding='utf-8',index=False)


    from sklearn import linear_model
    try:
        classifier = pickle.load(open('SGDTest.sav','rb'))
        print('\nModel loaded')
        flag = 1
    except:
        classifier = linear_model.SGDClassifier()
        print('Model Created')
        flag = 0
    

    
    if flag == 0:
        print("Model first fit commencing...")
        classifier.fit(X,y)    
        pickle.dump(classifier,open('SGDTest.sav','wb'))
        print("Model first fit successful !")

    
    predict_data(classifier,scalar)