# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:29:59 2020

@author: swpan
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import pickle
from termcolor import colored
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn import linear_model,naive_bayes

global classifier

flag = -1

model_name = 'PassiveAggressiveClassifier.sav'

try:
    #classifier = pickle.load(open('SGDClassifiermodel.sav','rb'))
    classifier = pickle.load(open(model_name,'rb'))
    print('\nModel loaded')
    flag = 1
except:
    classifier = linear_model.PassiveAggressiveClassifier()
    #classifier = naive_bayes.BernoulliNB()
    print('Model Created')
    flag = 0

def Feedback(df):
    
    global classifier
    
    x_new = df.iloc[:,2:-1].values
    y_new = df.iloc[:,-1].values
    y_new = y_new.astype(int)
    
    #scalar = MinMaxScaler().fit(x_new)
    #x_new = scalar.transform(x_new) 
    print('partial fitting the model')
    classifier.partial_fit(x_new,y_new)
   
    pickle.dump(classifier,open(model_name,'wb'))
    print('new model saved')
    

def AddNewPatient(dataset,scalar,classifier):
    
    print("\n-- Please add the following details of the patient -- ")
    
    id = dataset.tail(1)['id'].values[0] + 1
    
    gender = input("Gender : ")
    if len(gender) < 1 : gender = "Male"
    else:
        gender = gender.lower()
        if gender.startswith("f"):
            gender = "Female"
        else:
            gender = "Female"
            
    age = input("AGE : ")
    if len(age) < 1 : age = 50
    else : age = int(age)
    
    fbs = input("FBS : ")
    if len(fbs) < 1 : fbs = 250
    else : fbs = int(fbs)
    
    pp2bs = input("PP2BS : ")
    if len(pp2bs) < 1 : pp2bs = 300
    else : pp2bs = int(pp2bs)
    
    creatinine = input("CREATININE : ")
    if len(creatinine) < 1 : creatinine = 1.2
    else : creatinine = float(creatinine)
    
    hba1c = input("HBA1C : ")
    if len(hba1c) < 1 : hba1c = 4.3
    else : hba1c = float(hba1c)
    
    urine = input("URINE KETOACIDOSIS (0/1) : ")
    if len(urine) < 1 : urine = 0
    else : urine = int(urine)
    
    fundus = input("FUNDUS RETINOTHERAPY (0/1) : ")
    if len(fundus) < 1 : fundus = 0
    else : fundus = int(fundus)
    
    fbs_normal = input("FBS (MEDIAN) : ")
    if len(fbs_normal) < 1 : fbs_normal = 180
    else : fbs_normal = int(fbs_normal)
    
    pp2bs_normal = input("PP2BS (MEDIAN) : ")
    if len(pp2bs_normal) < 1 : pp2bs_normal = 160
    else : pp2bs_normal = int(pp2bs_normal)
    
    days_since_last_visit = input("Days since last visit : ")
    if len(days_since_last_visit) < 1 : days_since_last_visit = 70
    else : days_since_last_visit = int(days_since_last_visit)
    
    dataset = dataset.append({'id':id,'gender':gender,'age':age,'fbs':fbs,'pp2bs':pp2bs,
                              'creatinine':creatinine,'hba1c':hba1c,'urine_ketoacidosis':urine,
                              'fundus_retinotherapy':fundus,'fbs_normal':fbs_normal,
                              'pp2bs_normal':pp2bs_normal,'days_from_last_appointment':days_since_last_visit,
                              'critical':0},ignore_index=True)
    
    temp = dataset.tail(1).values
    x_temp_new = temp[:,2:-1]
    x_temp_new = scalar.transform(x_temp_new)
    
    predicted = int(classifier.predict(x_temp_new)[0])
    dataset.iloc[-1, dataset.columns.get_loc('critical')] = predicted
    temp = dataset.tail(1).values
    y_temp_new = temp[:,-1]
    
    
    if predicted == 1:
        print(colored('Patient is Critical !', 'red',attrs=['bold']))
        #print("Patient is Critical !")
    else:
        print(colored('Patient is NOT Critical !', 'green',attrs=['bold']))
        #print("Patient is not Critical !")
    
    ans = input("Do you agree ? (y/n): ")
    
    if ans == 'y' or ans == 'Y' or ans == 1:
        dataset.to_csv('test_data.csv',encoding='utf-8',index=False)
        classifier.partial_fit(x_temp_new,y_temp_new)
        write_to_doc_answers(predicted)
        
    else:
        if predicted == 1:
            #dataset.tail(1)['critical'] = 0
            dataset.iloc[-1, dataset.columns.get_loc('critical')] = 0
            write_to_doc_answers(0)
        else:
            dataset.iloc[-1, dataset.columns.get_loc('critical')] = 1
            write_to_doc_answers(1)
            
        temp = dataset.tail(1).values
        y_temp_new = temp[:,-1]
        dataset.to_csv('test_data.csv',encoding='utf-8',index=False)
        classifier.partial_fit(x_temp_new,y_temp_new)
       
    f = open("doc_ans.txt")
    for line in f:
        line.rstrip()
        doc_ans = line.split(',')
    f.close()
    dataset1 = pd.read_csv('test_data.csv')
    finaldata = dataset1.iloc[:,:].values
    retrain_model(finaldata,doc_ans,classifier)
             
    return dataset
    

def write_to_doc_answers(data):
    f1 = open("doc_ans.txt", "a")
    str1 = ","+str(data)
    f1.write(str1)
    f1.close()
    
    
def retrain_model(finaldata,docs_ans,classifier):
    count = 0
    print("\nCommencing model fitting...")
    
    prev = []
    ids_count = 0
    while True:
        nextids = []
        newcount = 0
        if flag != 0 and count != 0:
            classifier = pickle.load(open(model_name,'rb'))
        columns=['id','gender','age','fbs','pp2bs','creatinine','hba1c','urine_ketoacidosis','fundus_retinotherapy','fbs_normal','pp2bs_normal','days_from_last_appointment','critical']
        df=pd.DataFrame(columns=columns) 
        for i in finaldata:
            x_temp = [i[2:-1]]
            x_temp = scalar.transform(x_temp) 
            y_temp = classifier.predict(x_temp)
            
            if(int(doc_ans[newcount])!=y_temp[0]):
                df=df.append({'id':i[0],'gender':i[1],'age':X[newcount][0],'fbs':X[newcount][1],'pp2bs':X[newcount][2],'creatinine':X[newcount][3],'hba1c':X[newcount][4],'urine_ketoacidosis':X[newcount][5],'fundus_retinotherapy':X[newcount][6],'fbs_normal':X[newcount][7],'pp2bs_normal':X[newcount][8],'days_from_last_appointment':X[newcount][9],'critical':doc_ans[newcount]},ignore_index=True)
                nextids.append(i[0])
                if count == 0:
                    prev.append(i[0])
            newcount+=1
            
        if df.shape[0] > 0:
            if count !=0:
                if prev == nextids:
                    #print("Equal")
                    ids_count += 1
                else:
                    #print("Not Equal")
                    prev = nextids
                    ids_count = 0
                    
            
            print("Repeat Count : "+str(ids_count))
            
            """
            if ids_count > 2:
                temp = ids_count/2    
                p = 0    
                for x1 in range(int(temp)):
                    df = df.append(df)
                    df = df.append(df)
                    df = df.append(df)
                    df = df.append(df)
                    df = df.append(df)
                    df = df.append(df)
                    
                    if p == x1:
                        df = df.append(df)
                        
                        if x1 == 0:
                            p += 1
                        else:
                            p += int(x1)"""
            
            Feedback(df)
            count+=1
            print(str(count)," | Mismatches -> ",str(df.shape[0]))
            
        else:
            if count > 0:
                print('\nModel Fitting Complete !')
            else:
                print("\nModel Already Up to Date !")
            break

def retrain_model1(classifier):
    
    dataset = pd.read_csv('test_data.csv')
    finaldata = dataset.iloc[:,:].values
    f = open("doc_ans.txt")
    
    for line in f:
        line.rstrip()
        doc_ans = line.split(',')
    f.close()
    count = 0
    print("\nCommencing model fitting...")
    while True:
        newcount = 0
        if flag != 0 and count != 0:
            classifier = pickle.load(open(model_name,'rb'))
        columns=['id','gender','age','fbs','pp2bs','creatinine','hba1c','urine_ketoacidosis','fundus_retinotherapy','fbs_normal','pp2bs_normal','days_from_last_appointment','critical']
        df=pd.DataFrame(columns=columns) 
        for i in finaldata:
            x_temp = [i[2:-1]]
            x_temp = scalar.transform(x_temp) 
            y_temp = classifier.predict(x_temp)
            
            if(int(doc_ans[newcount])!=y_temp[0]):
                df=df.append({'id':i[0],'gender':i[1],'age':X[newcount][0],'fbs':X[newcount][1],'pp2bs':X[newcount][2],'creatinine':X[newcount][3],'hba1c':X[newcount][4],'urine_ketoacidosis':X[newcount][5],'fundus_retinotherapy':X[newcount][6],'fbs_normal':X[newcount][7],'pp2bs_normal':X[newcount][8],'days_from_last_appointment':X[newcount][9],'critical':doc_ans[newcount]},ignore_index=True)
                
            newcount+=1
            
        if df.shape[0] > 0:
            Feedback(df)
            count+=1
            print(str(count)," | Mismatches -> ",str(df.shape[0]))
            
        else:
            print('\nModel Fitting Complete !')
            break

def TestPatient(dataset,scalar):
    
    x_temp_new = dataset[:,2:-1]
    x_temp_new = scalar.transform(x_temp_new)
    
    predicted = int(classifier.predict(x_temp_new)[0])
    
    print("----------------\nID : "+str(dataset[0][0]))
    
    print("GENDER : "+str(dataset[0][1])+
          "\nAGE : "+str(dataset[0][2])+
          "\nFBS : "+str(dataset[0][3])+
          "\nPP2BS : "+str(dataset[0][4])+
          "\nCREATININE : "+str(dataset[0][5])+
          "\nHBA1C : "+str(dataset[0][6])+
          "\nURINE KETOACIDOSIS : "+str(dataset[0][7])+
          "\nFUNDUS RETINOTHERAPY : "+str(dataset[0][8])+
          "\nDAYS FROM LAST VISIT : "+str(dataset[0][11])
          )
    
    if predicted == 1:
        print(colored('Patient is Critical !', 'red',attrs=['bold']))
        #print("Patient is Critical !")
    else:
        print(colored('Patient is NOT Critical !', 'green',attrs=['bold']))
        #print("Patient is not Critical !")
    print("----------------")

if __name__=='__main__':
    
    #print(str(flag))
    
    dataset = pd.read_csv('test_data.csv')
    
    df = pd.read_csv('critical_data.csv')
    
    m = dataset.iloc[:,2:-1].values
    y = dataset.iloc[:, -1].values
        
    scalar = MinMaxScaler().fit(m)
    X= scalar.transform(m) 
    #print(X)
    #print(y)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
    
    if flag == 0:
        classifier.fit(X,y)
        pickle.dump(classifier,open(model_name,'wb'))
        print("Model first training successful !")
    
    #a = [[40,130,140,0.2,5.0,0,0,120,148,44]]
    
    #a=scalar.transform(a)
    
    #b=classifier.predict(a)
    
    finaldata = dataset.iloc[:,:].values
    columns=['id','gender','age','fbs','pp2bs','creatinine','hba1c','urine_ketoacidosis','fundus_retinotherapy','fbs_normal','pp2bs_normal','days_from_last_appointment','critical']
    df=pd.DataFrame(columns=columns) 
    count=1
    newcount=0
    
    try:
        f= open("doc_ans.txt")
        
        for line in f:
            line.rstrip()
            doc_ans = line.split(',')
     
        f.close()
    except:
        print("Doctor answers loading failed")
        
    #doc_ans = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    
    """
    for i in finaldata:
        
        x_temp = [i[2:-1]]
        
        
        x_temp = scalar.transform(x_temp) 
        
        y_temp = classifier.predict(x_temp)
        
        print(str(count)+")")
        
        print("\tID : "+str(i[0]))
        print("\tGender : "+str(i[1]))
        print("\tAge : "+str(i[2]))
        
        fbs_level = "(Normal)"
        
        fbs_diff = abs(i[3]-i[9])
        if fbs_diff > 40:
            if i[3]>140 :
                fbs_level = "(High)"
            elif i[3] < 80:
                fbs_level = "(Low - Hypoglycemia)"
            
            
        print("\tFBS : "+str(i[3])+" "+fbs_level)
        print("\tFBS (Median): "+str(i[9]))
        
        
        pp2bs_level = "(Normal)"
        
        pp2bs_diff = abs(i[4]-i[10])
        if pp2bs_diff > 40:
            if i[4]>140 :
                pp2bs_level = "(High)"
            elif i[4] < 80:
                pp2bs_level = "(Low - Hypoglycemia)"
        
        
        print("\tPP2BS : "+str(i[4])+" "+pp2bs_level)
        print("\tPP2BS (Median): "+str(i[10]))
        
        creatinine_level = '(Normal)'
        if(i[5]>1.5):
            creatinine_level='(High)'
        
        
        print("\tCREATININE : "+str(i[5])+' '+creatinine_level)
        
        hba1c_level = '(Normal)'
        if(i[6]>7):
            hba1c_level='(High)'
             
        print("\tHBA1C : "+ str(i[6])+' '+ hba1c_level)
            
        
        print("\tURINE KETOACIDOSIS : "+str(i[7]))
        print("\tFUNDUS RETINOTHERAPY : "+str(i[8]))
        print("\tDAYS SINCE LAST VISIT : "+str(i[11]))
        print("\tCritical : "+str(y_temp[0]))
        
        print("\n\n")
        
        count+=1
        
        n = int(input("Do you want to call this patient? Please Ans in 0/1 : "))
        doc_ans.append(n)
       
        if(n!=y_temp[0]):
            #df = df[i]
            
            df=df.append({'id':i[0],'gender':i[1],'age':X[newcount][0],'fbs':X[newcount][1],'pp2bs':X[newcount][2],'creatinine':X[newcount][3],'hba1c':X[newcount][4],'urine_ketoacidosis':X[newcount][5],'fundus_retinotherapy':X[newcount][6],'fbs_normal':X[newcount][7],'pp2bs_normal':X[newcount][8],'days_from_last_appointment':X[newcount][9],'critical':n},ignore_index=True)
        newcount+=1
        
    """
    
    """
    count = 0
    print("\nCommencing model fitting...")
    while True:
        newcount = 0
        if flag != 0 and count != 0:
            classifier = pickle.load(open('SGDClassifiermodel.sav','rb'))
        columns=['id','gender','age','fbs','pp2bs','creatinine','hba1c','urine_ketoacidosis','fundus_retinotherapy','fbs_normal','pp2bs_normal','days_from_last_appointment','critical']
        df=pd.DataFrame(columns=columns) 
        for i in finaldata:
            x_temp = [i[2:-1]]
            x_temp = scalar.transform(x_temp) 
            y_temp = classifier.predict(x_temp)
            
            if(int(doc_ans[newcount])!=y_temp[0]):
                df=df.append({'id':i[0],'gender':i[1],'age':X[newcount][0],'fbs':X[newcount][1],'pp2bs':X[newcount][2],'creatinine':X[newcount][3],'hba1c':X[newcount][4],'urine_ketoacidosis':X[newcount][5],'fundus_retinotherapy':X[newcount][6],'fbs_normal':X[newcount][7],'pp2bs_normal':X[newcount][8],'days_from_last_appointment':X[newcount][9],'critical':doc_ans[newcount]},ignore_index=True)
                
            newcount+=1
            
        if df.shape[0] > 0:
            Feedback(df)
            count+=1
            print(str(count)," | Mismatches -> ",str(df.shape[0]))
            
        else:
            print('\nModel Fitting Complete !')
            break
    """
    
    
    
    while True:
        try:
            print("\n0. Exit\n1. Add Patient\n2. Test Last Patient\n3. Test Patient by ID\n4. Retrain the Model")
            menu_option = int(input("Option : "))
        except:
            print("Please enter valid input !")
        
        if menu_option == 0:
            break
        elif menu_option == 1:
            AddNewPatient(dataset,scalar,classifier)
            dataset = pd.read_csv('test_data.csv') 
        elif menu_option == 2 :
            TestPatient(dataset.tail(1).values,scalar)
        elif menu_option == 3:
            try:
                patient_id = int(input("Patient ID : ")) 
                #print(type(patient_id),type(dataset.shape[0]))
                if patient_id >= dataset.shape[0] or patient_id < 0:
                    print("Patient ID Out of Bounds !")
                else:
                    patient = dataset[dataset['id'] == patient_id]
                    TestPatient(patient.values,scalar)
            except Exception as e:
                print("Invalid patient ID ! : " + str(e))
        elif menu_option == 4:
            f = open("doc_ans.txt")
            for line in f:
                line.rstrip()
                doc_ans = line.split(',')
            f.close()
            retrain_model(finaldata,doc_ans,classifier)
                
                