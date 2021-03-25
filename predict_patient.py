import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


data = {'fbs_high' : [0],'fbs_normal' : [0],'fbs_low':[0],'pp2bs_high' : [0],'pp2bs_normal' : [0],'pp2bs_low':[0],'creatinine' : [0],'hba1c' : [0],'urine' : [0],'fundus' : [0],'none':[0]}

df = pd.DataFrame(data) 


 #Limits
fbs_limits = [80,140,400]
hba1c_pre_limit = [6,7]
creatinine_serum_limit = 1.5

weight_csv = pd.read_csv('weights.csv')

#Weights
fbs_weight_high = weight_csv['fbs_weight_high'].values[0]
fbs_weight_normal = weight_csv['fbs_weight_normal'].values[0]
fbs_weight_low = weight_csv['fbs_weight_low'].values[0]
pp2bs_weight_high = weight_csv['pp2bs_weight_high'].values[0]
pp2bs_weight_normal = weight_csv['pp2bs_weight_normal'].values[0]
pp2bs_weight_low = weight_csv['pp2bs_weight_low'].values[0]
hba1c_pre_weight = weight_csv['hba1c_pre_weight'].values[0]
hba1c_dia_weight = weight_csv['hba1c_dia_weight'].values[0]
fundus_weight = weight_csv['fundus_weight'].values[0]
creatinine_serum_weight = weight_csv['creatinine_serum_weight'].values[0]
ketoacidosis_weight = weight_csv['ketoacidosis_weight'].values[0]


def load_original_weights():
    global fbs_weight_high 
    global fbs_weight_low
    global fbs_weight_normal
    global pp2bs_weight_high 
    global pp2bs_weight_low
    global pp2bs_weight_normal
    global hba1c_pre_weight
    global hba1c_dia_weight
    global fundus_weight
    global creatinine_serum_weight
    global ketoacidosis_weight
    global weight_csv
    weight_csv = pd.read_csv('weightsoriginal.csv')
        
    #Weights
    fbs_weight_high = weight_csv['fbs_weight_high'].values[0]
    fbs_weight_normal = weight_csv['fbs_weight_normal'].values[0]
    fbs_weight_low = weight_csv['fbs_weight_low'].values[0]
    pp2bs_weight_high = weight_csv['pp2bs_weight_high'].values[0]
    pp2bs_weight_normal = weight_csv['pp2bs_weight_normal'].values[0]
    pp2bs_weight_low = weight_csv['pp2bs_weight_low'].values[0]
    hba1c_pre_weight = weight_csv['hba1c_pre_weight'].values[0]
    hba1c_dia_weight = weight_csv['hba1c_dia_weight'].values[0]
    fundus_weight = weight_csv['fundus_weight'].values[0]
    creatinine_serum_weight = weight_csv['creatinine_serum_weight'].values[0]
    ketoacidosis_weight = weight_csv['ketoacidosis_weight'].values[0]
    
    weight_csv.to_csv('weights.csv',encoding='utf-8',index=False)
        
    

def calculate_fbs_risk(fbs,fbs_normal):
    evaluation = 0
    fbs_diff = abs(fbs-fbs_normal)
    if fbs_diff>40:
        if fbs < fbs_limits[0]:
            evaluation += fbs_weight_low - (fbs/1000)
        elif fbs >= fbs_limits[0] and fbs <= fbs_limits[1]:
            evaluation += 0
        elif fbs > fbs_limits[1] and fbs < fbs_limits[2] :
            evaluation += fbs_weight_normal +(fbs/1000)
        else:
            evaluation += fbs_weight_high +(fbs/100)
            
    return evaluation
    
    
def calculate_pp2bs_risk(pp2bs,pp2bs_normal):
    evaluation = 0
    pp2bs_diff = abs(pp2bs-pp2bs_normal)    
    if pp2bs_diff>40:
        if pp2bs < fbs_limits[0]:
            evaluation += pp2bs_weight_low - (pp2bs/1000)
        elif pp2bs >= fbs_limits[0] and pp2bs <= fbs_limits[1]:
            evaluation += 0
        elif pp2bs > fbs_limits[1] and pp2bs < fbs_limits[2] :
            evaluation += pp2bs_weight_normal +(pp2bs/1000)
        else:
            evaluation += pp2bs_weight_high +(pp2bs/100)
    return evaluation

def calculate_hba1c_risk(hba1c):
    evaluation = 0
    if hba1c >= hba1c_pre_limit[0] and hba1c <= hba1c_pre_limit[1]:
        evaluation += hba1c_pre_weight + (hba1c/100)
    elif  hba1c >= hba1c_pre_limit[1]:
        evaluation += hba1c_dia_weight + (hba1c/100)
    
    return evaluation

def calculate_fundus_risk(fundus_retinotherapy):
    evaluation = 0
    if fundus_retinotherapy == 1:
        evaluation +=  fundus_weight
    
    return evaluation


def calculate_creatinine_risk(creatinine):
    evaluation = 0
    if creatinine >= creatinine_serum_limit:
        evaluation += creatinine_serum_weight + (creatinine/10)
    return evaluation


def calculate_urine_risk(urine_ketoacidosis):
    evaluation = 0 
    if urine_ketoacidosis == 1:
        evaluation += ketoacidosis_weight
    
    return evaluation

def random_forest(dataset):
    

    #new_dataset = dataset[['fbs','pp2bs','creatinine','hba1c','urine_ketoacidosis','fundus_retinotherapy','age','days_from_last_appointment']]
    #dataset['risk-factor'] =  dataset.apply(dia.calculate_risk1,axis=1)
    #dataset.to_csv('diabetes_data_low_with_risk.csv',encoding='utf-8',index=False)
    
    
    
    #print(dataset.head())
    
    #df['equal_or_lower_than_4?'] = df['set_of_numbers'].apply(lambda x: 'True' if x <= 4 else 'False')
    
    #dataset['urine-for-ketoacidosis'] = dataset['urine-for-ketoacidosis'].apply(truefalse)
    #dataset['fundus-retinotherapy'] = dataset['fundus-retinotherapy'].apply(truefalse)
    #dataset.to_csv('diabetes_data.csv',encoding='utf-8',index=False)
    #mymap = {"true":1, "false":0}
    #dataset.applymap(lambda s: mymap.get(s) if s in mymap else s)
    #print(dataset.head())
    #print(type(dataset['urine-for-ketoacidosis'][0]))
    m = dataset.iloc[:,2:-1].values
    y = dataset.iloc[:, -1].values
        
    scalar = StandardScaler().fit(m)
    X= scalar.transform(m) 
    #print(X)
    #print(y)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
    
    #from sklearn.linear_model import LinearRegression
    #regressor = LinearRegression()
    from sklearn.ensemble import RandomForestRegressor
    regressor=RandomForestRegressor(n_estimators = 50, random_state = 0)
    regressor.fit(X_train, y_train)
    
    pickle.dump(regressor,open('randomforestmodel.sav','wb'))
    
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    print(X_test)
    
    
    print("\nTrue Output \n")
    print(y_test)
    
    print("\nPredicted Output \n")
    print(y_pred)
    
    
    print(type(y_pred))
    print("\n mean square error")
    
    from sklearn.metrics import mean_squared_error
    print(mean_squared_error(y_test, y_pred))
    
    
    
    
    # Visualising the Random Forest Regression results (higher resolution)
    #X_grid = np.arange(min(X), max(X), 1)
    #X_grid = X_grid.reshape((len(X_grid), 1))
    k=np.arange(1,19441,200)
    #plt.scatter(k, y_test, color = 'red')
    y_test_new = y_test[k] 
    
    
    y_pred_new = y_pred[k] 
    plt.plot(k, y_test_new, '-b',label='test data')
    plt.plot(k, y_pred_new, '-r',label='predicted data')
    plt.title('Random forest')
    plt.xlabel('patients')
    plt.ylabel('risk factor')
    plt.legend(loc="upper left")
    plt.show()
    
    error=0
    for i in range(y_test.size):
        diff = abs(y_test[i]-y_pred[i])
        if diff>1:
            error+=1
    
    h = y_test.size
    
    acc = ((h - error)/h)*100


def weight_reduction(rejected, total,alldata,flag_has_upvoted):
    global fbs_weight_high 
    global fbs_weight_low
    global fbs_weight_normal
    global pp2bs_weight_high 
    global pp2bs_weight_low
    global pp2bs_weight_normal
    global hba1c_pre_weight
    global hba1c_dia_weight
    global fundus_weight
    global creatinine_serum_weight
    global ketoacidosis_weight
    global weight_csv
    global df
    
    total_sum = df.sum(axis=1).values[0]
    multiplier1 = (rejected/total)
    
    df = 1 -  (df/total_sum)*multiplier1;
    fbs_weight_high *= df['fbs_high'].values[0]
    fbs_weight_low *= df['fbs_low'].values[0]
    fbs_weight_normal *= df['fbs_normal'].values[0]
    pp2bs_weight_high *= df['pp2bs_high'].values[0]
    pp2bs_weight_low *= df['pp2bs_low'].values[0]
    pp2bs_weight_normal *= df['pp2bs_normal'].values[0]
    hba1c_pre_weight *= df['hba1c'].values[0]
    hba1c_dia_weight *= df['hba1c'].values[0]
    ketoacidosis_weight *= df['urine'].values[0]
    fundus_weight *= df['fundus'].values[0]
    
    weight_csv['fbs_weight_high']=fbs_weight_high
    weight_csv['fbs_weight_low']=fbs_weight_low
    weight_csv['fbs_weight_normal']=fbs_weight_normal
    weight_csv['pp2bs_weight_high']=pp2bs_weight_high
    weight_csv['pp2bs_weight_low']=pp2bs_weight_low
    weight_csv['pp2bs_weight_normal']=pp2bs_weight_normal
    weight_csv['hba1c_pre_weight']=hba1c_pre_weight
    weight_csv['hba1c_dia_weight']=hba1c_dia_weight
    weight_csv['ketoacidosis_weight']=ketoacidosis_weight
    weight_csv['fundus_weight']=fundus_weight
    
    weight_csv.to_csv('weights.csv',encoding='utf-8',index=False)
    
    
    if flag_has_upvoted == 0:
        
        """
        alldata = calculate_newrisk()
        random_forest(alldata)
        """
    
    
    #if df['fbs'] != 0:
    
def weight_increament(rejected, total,alldata):
    global fbs_weight_high 
    global fbs_weight_low
    global fbs_weight_normal
    global pp2bs_weight_high 
    global pp2bs_weight_low
    global pp2bs_weight_normal
    global hba1c_pre_weight
    global hba1c_dia_weight
    global fundus_weight
    global creatinine_serum_weight
    global ketoacidosis_weight
    global weight_csv
    global df
    
    total_sum = df.sum(axis=1).values[0]
    multiplier1 = (rejected/total)
    
    total_sum = df.sum(axis=1).values[0]
    multiplier1 = (rejected/total)
    
    df = 1 +  (df/total_sum)*multiplier1;
    fbs_weight_high *= df['fbs_high'].values[0]
    fbs_weight_low *= df['fbs_low'].values[0]
    fbs_weight_normal *= df['fbs_normal'].values[0]
    pp2bs_weight_high *= df['pp2bs_high'].values[0]
    pp2bs_weight_low *= df['pp2bs_low'].values[0]
    pp2bs_weight_normal *= df['pp2bs_normal'].values[0]
    hba1c_pre_weight *= df['hba1c'].values[0]
    hba1c_dia_weight *= df['hba1c'].values[0]
    ketoacidosis_weight *= df['urine'].values[0]
    fundus_weight *= df['fundus'].values[0]
    
    weight_csv['fbs_weight_high']=fbs_weight_high
    weight_csv['fbs_weight_low']=fbs_weight_low
    weight_csv['fbs_weight_normal']=fbs_weight_normal
    weight_csv['pp2bs_weight_high']=pp2bs_weight_high
    weight_csv['pp2bs_weight_low']=pp2bs_weight_low
    weight_csv['pp2bs_weight_normal']=pp2bs_weight_normal
    weight_csv['hba1c_pre_weight']=hba1c_pre_weight
    weight_csv['hba1c_dia_weight']=hba1c_dia_weight
    weight_csv['ketoacidosis_weight']=ketoacidosis_weight
    weight_csv['fundus_weight']=fundus_weight
    
    weight_csv.to_csv('weights.csv',encoding='utf-8',index=False)
    
    
    
    #trains random forest model using 1.9 lac data
    """
    alldata = calculate_newrisk()
    random_forest(alldata)
    """
    

def calculate_risk1(row):
    evaluation = 0
    
    evaluation += calculate_fbs_risk(row['fbs'],row['fbs_normal'])
    
    evaluation += calculate_fbs_risk(row['pp2bs'],row['pp2bs_normal'])
        
    evaluation += calculate_hba1c_risk(row['hba1c'])

    evaluation += calculate_fundus_risk(row['fundus_retinotherapy'])

    evaluation += calculate_creatinine_risk(row['creatinine'])

    evaluation += calculate_urine_risk(row['urine_ketoacidosis'])
   
    if row['age'] >1:
        if row['age']<20:
            evaluation *= 1.1
        elif row['age']<40:
            evaluation *= 1.2
        elif row['age']<60:
            evaluation *= 1.3
        else:
            evaluation *= 1.4
    
    if row['days_from_last_appointment'] > 0:
        if row['days_from_last_appointment']>90:
            evaluation *=2
        else:
            x = row['days_from_last_appointment']/90
            evaluation *= (x+1)
    
    print(str(row['id'])+" -> "+str(evaluation))

    return evaluation


def calculate_newrisk():
    newdf = pd.read_csv('diabetes_data_2.7_with_risk.csv')
    #newdf = pd.read_csv('.csv')
    newdf['risk-factor'] =  newdf.apply(calculate_risk1,axis=1)
    return newdf


def predcit1():

    loaded_model = pickle.load(open('randomforestmodel.sav','rb'))
    
    dataset = pd.read_csv('test_data.csv')
    
    
    dataset['risk-factor-predicted'] = loaded_model.predict(dataset.iloc[:, 5:].values);
    
    dataset = dataset.sort_values('risk-factor-predicted',ascending=False)
    
    #result = loaded_model.predict()
    
    
def feedback(patient_count,patient_data):
    
    global df
    global fbs_limits
    
    rejected = [] 
   
    n = input("Enter patient ID's rejected (-1 for none) : ") 
    
    flag_has_rejected = 0
    if n != '-1':
        flag_has_rejected = 1
    
    upvote_ids = input("Enter patient ID's to upvote (-1 for none) : ") 
    
    flag_has_upvoted = 0
    if upvote_ids != '-1':
        flag_has_upvoted = 1
        
    print("\n\n")
    
    rejected = n.split(' ')
    
    if flag_has_rejected != 0:
        
        for k in rejected:
            reasons = []
            #print(str(i))
            
            i = int(k)
            risk = patient_data[patient_data['id'] ==i]['risk-factor-predicted'].values[0]
            
            percentages = {'FBS':[0],'PP2BS':[0],'CREATININE':[0],'HBA1C':[0],'URINE-KETOACIDOSIS':[0],'FUNDUS-RETINOTHERAPY':[0]}
            
            #fbs percent
            fbs_temp = patient_data[patient_data['id'] == i]['fbs'].values[0]
            fbs_percent = (calculate_fbs_risk(fbs_temp,patient_data[patient_data['id'] == i]['fbs_normal'].values[0] )/risk)*100
            percentages['FBS'] = fbs_percent
            
            #pp2bs percent
            pp2bs_temp = patient_data[patient_data['id'] == i]['pp2bs'].values[0]
            pp2bs_percent = (calculate_pp2bs_risk(pp2bs_temp,patient_data[patient_data['id'] == i]['pp2bs_normal'].values[0] )/risk)*100
            percentages['PP2BS'] = pp2bs_percent
            
            #HBA1C
            hba1c_temp = patient_data[patient_data['id'] == i]['hba1c'].values[0]
            hba1c_percent = (calculate_hba1c_risk(hba1c_temp)/risk)*100
            percentages['HBA1C'] = hba1c_percent
            
            #Creatinine
            creatinine_temp = patient_data[patient_data['id'] == i]['creatinine'].values[0]
            percentages['CREATININE'] = (calculate_creatinine_risk(creatinine_temp)/risk)*100
            
        
            #Urine-Ketoacidosis
            urine_temp = patient_data[patient_data['id'] == i]['urine_ketoacidosis'].values[0]
            percentages['URINE-KETOACIDOSIS'] = (calculate_urine_risk(urine_temp)/risk)*100
            
            
            #Fundus-Retinotherapy
            fundus_temp = patient_data[patient_data['id'] == i]['fundus_retinotherapy'].values[0]
            percentages['FUNDUS-RETINOTHERAPY'] = (calculate_fundus_risk(fundus_temp)/risk)*100
            
            percentages = sorted(percentages.items(), key = lambda kv:(kv[1], kv[0]),reverse = True)
            
            #print(percentages)
            
            #Print Percentages 
            print("\nFor Patient ID : ",k)
            for percent in percentages:
                if percent[1] > 0:
                    print(percent[0]," : ",round(percent[1], 2),"%")
            
        
            print("\nReason for rejection ? \n 1.FBS \n 2.PP2BS \n 3.CREATININE \n 4.HBA1C \n 5.URINE KETOACIDOCIS \n 6.FUNDUS RETINOTHERAPY \n 7. None \n");
            
            
            
            x = input("params : ")
            reasons = x.split(' ')
            
            for j in reasons :
                if(int(j) == 1):
                    if fbs_temp < fbs_limits[0]:
                        df['fbs_low'] += 1
                    elif fbs_temp >= fbs_limits[1] and fbs_temp <= fbs_limits[2]:
                        df['fbs_normal'] +=1
                    else:
                        df['fbs_high'] +=1
                elif(int(j) == 2):
                    if pp2bs_temp < fbs_limits[0]:
                        df['pp2bs_low'] += 1
                    elif pp2bs_temp >= fbs_limits[1] and pp2bs_temp <= fbs_limits[2]:
                        df['pp2bs_normal'] +=1
                    else:
                        df['pp2bs_high'] +=1
                elif(int(j) == 3):
                    df['creatinine'] += 1
                elif(int(j) == 4):
                    df['hba1c'] += 1
                elif(int(j) == 5):
                    df['urine'] += 1
                elif(int(j) == 6):
                    df['fundus'] += 1
                elif(int(j) == 7):
                    df['none'] += 1
                    
        weight_reduction(len(rejected),patient_count,patient_data,flag_has_upvoted)
                                
                                
    
    data = {'fbs_high' : [0],'fbs_normal' : [0],'fbs_low':[0],'pp2bs_high' : [0],'pp2bs_normal' : [0],'pp2bs_low':[0],'creatinine' : [0],'hba1c' : [0],'urine' : [0],'fundus' : [0],'none':[0]}

    df = pd.DataFrame(data) 
    
    if flag_has_upvoted != 0:
        selected = upvote_ids.split(' ')
        
        for k in selected:
            reasons = []
            #print(str(i))
            
            i = int(k)
            risk = patient_data[patient_data['id'] ==i]['risk-factor-predicted'].values[0]
            
            percentages = {'FBS':[0],'PP2BS':[0],'CREATININE':[0],'HBA1C':[0],'URINE-KETOACIDOSIS':[0],'FUNDUS-RETINOTHERAPY':[0]}
            
            #fbs percent
            fbs_temp = patient_data[patient_data['id'] == i]['fbs'].values[0]
            fbs_percent = (calculate_fbs_risk(fbs_temp,patient_data[patient_data['id'] == i]['fbs_normal'].values[0] )/risk)*100
            percentages['FBS'] = fbs_percent
            
            #pp2bs percent
            pp2bs_temp = patient_data[patient_data['id'] == i]['pp2bs'].values[0]
            pp2bs_percent = (calculate_pp2bs_risk(pp2bs_temp,patient_data[patient_data['id'] == i]['pp2bs_normal'].values[0] )/risk)*100
            percentages['PP2BS'] = pp2bs_percent
            
            #HBA1C
            hba1c_temp = patient_data[patient_data['id'] == i]['hba1c'].values[0]
            hba1c_percent = (calculate_hba1c_risk(hba1c_temp)/risk)*100
            percentages['HBA1C'] = hba1c_percent
            
            #Creatinine
            creatinine_temp = patient_data[patient_data['id'] == i]['creatinine'].values[0]
            percentages['CREATININE'] = (calculate_creatinine_risk(creatinine_temp)/risk)*100
            
        
            #Urine-Ketoacidosis
            urine_temp = patient_data[patient_data['id'] == i]['urine_ketoacidosis'].values[0]
            percentages['URINE-KETOACIDOSIS'] = (calculate_urine_risk(urine_temp)/risk)*100
            
            
            #Fundus-Retinotherapy
            fundus_temp = patient_data[patient_data['id'] == i]['fundus_retinotherapy'].values[0]
            percentages['FUNDUS-RETINOTHERAPY'] = (calculate_fundus_risk(fundus_temp)/risk)*100
            
            percentages = sorted(percentages.items(), key = lambda kv:(kv[1], kv[0]),reverse = True)
            
            #print(percentages)
            
            #Print Percentages 
            print("\nFor Patient ID : ",k)
            for percent in percentages:
                if percent[1] > 0:
                    print(percent[0]," : ",round(percent[1], 2),"%")
            
        
            print("\nReason for upvoting ? \n 1.FBS \n 2.PP2BS \n 3.CREATININE \n 4.HBA1C \n 5.URINE KETOACIDOCIS \n 6.FUNDUS RETINOTHERAPY \n 7. None \n");
            
            
            
            x = input("params : ")
            reasons = x.split(' ')
            
            for j in reasons :
                if(int(j) == 1):
                    if fbs_temp < fbs_limits[0]:
                        df['fbs_low'] += 1
                    elif fbs_temp >= fbs_limits[1] and fbs_temp <= fbs_limits[2]:
                        df['fbs_normal'] +=1
                    else:
                        df['fbs_high'] +=1
                elif(int(j) == 2):
                    if pp2bs_temp < fbs_limits[0]:
                        df['pp2bs_low'] += 1
                    elif pp2bs_temp >= fbs_limits[1] and pp2bs_temp <= fbs_limits[2]:
                        df['pp2bs_normal'] +=1
                    else:
                        df['pp2bs_high'] +=1
                elif(int(j) == 3):
                    df['creatinine'] += 1
                elif(int(j) == 4):
                    df['hba1c'] += 1
                elif(int(j) == 5):
                    df['urine'] += 1
                elif(int(j) == 6):
                    df['fundus'] += 1
                elif(int(j) == 7):
                    df['none'] += 1
        
        weight_increament(len(upvote_ids),patient_count,patient_data)  
                     
    print("\n\n")
     
    
     

if __name__ == "__main__":
    
    #df = calculate_newrisk(df)
    
    #random_forest(df)
    #load_original_weights();
    dataset = pd.read_csv('test_data.csv')
    
    
    #X = dataset.iloc[:, 2:].values
    
    #loaded_model = pickle.load(open('randomforestmodel.sav','rb'))
    
    #scalar = StandardScaler().fit(X)
    #X= scalar.transform(X) 
    
    # predict risk-factor using random forest model
    #dataset['risk-factor-predicted'] = loaded_model.predict(X);
    
    # predict risk-factor using python function
    dataset['risk-factor-predicted'] = dataset.apply(calculate_risk1,axis=1)
    
    dataset = dataset.sort_values('risk-factor-predicted',ascending=False)
    
    finaldata = dataset.iloc[:,:].values
    
    count2 = int(input("Number of patients you want to predict : "))
    
    count2 = count2*2
    count = 1;
    print("-------------------Next patients to call-------------------")
    for i in finaldata:
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
        print("\tRISK FACTOR : "+str(i[-1]))
        
        print("\n\n")
        if count == count2:
            break
        else:
            count += 1
        
        
       
        """
        will_call = input("WIll you call this patient?")
        if(will_call == '1' or will_call =="yes" or will_call == "y"):
            print("Why are you calling patient? \n 1.FBS \n 2.PP2BS \n 3.CREATININE \n 4.HBA1C \n 5.URINE KETOACIDOCIS \n 6.FUNDUS RETINOTHERAPY \n ");
            arr= list()
            arr = input("Give number of it")
            for i in arr:
                if(i==1):
                    a = 1
                    
          """      
        
    feedback(count2,dataset)
    
    
    #dataset['finalcall']= dataset['risk-factor-predicted'].apply(lambda x: 1 if x>42 else 0 )
    #dataset.to_csv('test_data2.csv',encoding='utf-8',index=False)
    #Sugar level plot
    # set width of bar
    barWidth = 0.25
    bar1 =  dataset.iloc[:5,3];
    bar2 = dataset.iloc[:5,4]
    ids = dataset.iloc[:5,0]
    # Set position of bar on X axis
    r1 = np.arange(len(bar1))
    r2 = [x + barWidth for x in r1]
    
    
    plt.bar(r1, bar1, color='#ec5051', width=barWidth, edgecolor='white', label='fbs')
    plt.bar(r2, bar2, color='#01a1ff', width=barWidth, edgecolor='white', label='pp2bs')
    
    plt.title("Sugar levels")
    # Add xticks on the middle of the group bars
    plt.xlabel('patient ids', fontweight='bold')
    plt.ylabel('sugar levels', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bar1))], ids)
    #plt.axhline(y=0.5, color='r', linestyle='-')
    
    # Create legend & Show graphic
    plt.legend()
    plt.show()
    
    
    # set width of bar
    barWidth = 0.25
    bar3 =  dataset.iloc[:5,5];
    bar4 = dataset.iloc[:5,6]
    ids = dataset.iloc[:5,0]
    # Set position of bar on X axis
    r1 = np.arange(len(bar3))
    r2 = [x + barWidth for x in r1]
    
    
    plt.bar(r1, bar3, color='#ec5051', width=barWidth, edgecolor='white', label='Creatinine')
    plt.bar(r2, bar4, color='#01a1ff', width=barWidth, edgecolor='white', label='hba1c')
    
    plt.title("Creatinine/ Hba1c levels")
    # Add xticks on the middle of the group bars
    plt.xlabel('patient ids', fontweight='bold')
    plt.ylabel('creatinine/hba1c', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bar3))], ids)
    #plt.axhline(y=0.5, color='r', linestyle='-')
    
    # Create legend & Show graphic
    plt.legend()
    plt.show()
