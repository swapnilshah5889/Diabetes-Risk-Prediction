import pandas as pd
import numpy as nm 
import json

 #Limits
fbs_limits = [80,140,400]
hba1c_pre_limit = [6,7]
creatinine_serum_limit = 1.5

#Weights
fbs_weight_high = 33
fbs_weight_normal = 1
fbs_weight_low = 33
hba1c_pre_weight = 2
hba1c_dia_weight = 3
fundus_weight = 4
creatinine_serum_weight = 5
ketoacidosis_weight = 16

def sugar_level(sugarlvl):
        if sugarlvl < fbs_limits[0]:
            return 'low'
        elif sugarlvl >= fbs_limits[0] and sugarlvl <= fbs_limits[1]:
            return 'normal'
        else:
            return 'high'

def top_patients(df,limit):
    return df[:limit]

def calculate_risk1(row):
    evaluation = 0
    fbs_diff = abs(row['fbs']-row['fbs_normal'])
    if fbs_diff>40:
        if row['fbs'] < fbs_limits[0]:
            evaluation += fbs_weight_low - (row['fbs']/100)
        elif row['fbs'] >= fbs_limits[0] and row['fbs'] <= fbs_limits[1]:
            evaluation += 0
        elif row['fbs'] > fbs_limits[1] and row['fbs'] < fbs_limits[2] :
            evaluation += fbs_weight_normal +(row['fbs']/1000)
        else:
            evaluation += fbs_weight_high +(row['fbs']/100)
    
    pp2bs_diff = abs(row['pp2bs']-row['pp2bs_normal'])    
    if pp2bs_diff>40:
        if row['pp2bs'] < fbs_limits[0]:
            evaluation += fbs_weight_low - (row['fbs']/100)
        elif row['pp2bs'] >= fbs_limits[0] and row['pp2bs'] <= fbs_limits[1]:
            evaluation += 0
        elif row['pp2bs'] > fbs_limits[1] and row['pp2bs'] < fbs_limits[2] :
            evaluation += fbs_weight_normal +(row['fbs']/1000)
        else:
            evaluation += fbs_weight_high +(row['pp2bs']/100)
        
    if row['hba1c'] >= hba1c_pre_limit[0] and row['hba1c'] <= hba1c_pre_limit[1]:
        evaluation += hba1c_pre_weight + (row['hba1c']/100)
    elif  row['hba1c'] >= hba1c_pre_limit[1]:
        evaluation += hba1c_dia_weight + (row['hba1c']/100)

    if row['fundus_retinotherapy'] == 1:
        evaluation +=  fundus_weight

    if row['creatinine'] >= creatinine_serum_limit:
        evaluation += creatinine_serum_weight + (row['creatinine']/10)
    
    if row['urine_ketoacidosis'] == 1:
        evaluation += ketoacidosis_weight
        
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



def calculate_risk(row):
    evaluation = 0
    if row['fbs'] < fbs_limits[0]:
       evaluation += fbs_weight_low - (row['fbs']/1000)
    elif row['fbs'] >= fbs_limits[0] and row['fbs'] <= fbs_limits[1]:
        evaluation += 0
    else:
        evaluation += fbs_weight_normal +(row['fbs']/1000)
    
    if row['pp2bs'] < fbs_limits[0]:
        evaluation += fbs_weight_low
    elif row['pp2bs'] >= fbs_limits[0] and row['pp2bs'] <= fbs_limits[1]:
        evaluation += 0
    else:
        evaluation += fbs_weight_normal
    
    if row['hba1c'] >= hba1c_pre_limit[0] and row['hba1c'] <= hba1c_pre_limit[1]:
       evaluation += hba1c_pre_weight + (row['hba1c']/100)
    elif  row['hba1c'] >= hba1c_pre_limit[1]:
        evaluation += hba1c_dia_weight + (row['hba1c']/100)

    if row['fundus-retinotherapy'] == 1:
        evaluation +=  fundus_weight

    if row['creatinine-serum'] >= creatinine_serum_limit:
       evaluation += creatinine_serum_weight + (row['creatinine-serum']/10)
    
    if row['urine-for-ketoacidosis'] == 1:
        evaluation += ketoacidosis_weight
    
    
    print(str(row['id'])+" -> "+str(evaluation))

    return evaluation


if __name__ == "__main__":

    
    diabetes_df = pd.read_csv('diabetes_data.csv',index_col=0)


    diabetes_df['risk-factor'] =  diabetes_df.apply(calculate_risk,axis=1) 
    diabetes_df = diabetes_df.sort_values(['risk-factor'],ascending = False)
    
    limit = int(input("Top limit : "))

    df1 = top_patients(diabetes_df,limit)
    df1 = df1.drop(['email','mobile-number','address','gender'],axis=1)
    jsondata = df1.to_json(orient='records', lines=True)
    
    #response = {'status':True,'total_records':limit,'data' : jsondata}
    
    print(jsondata)
    #print(diabetes_df['result'])
    #print(list(diabetes_df.columns.values))
    #print(diabetes_df['name'])
    #print(diabetes_df.tail())


