import pandas as pd
import numpy as np
import random
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
import diabetes as dia

creat_arr = np.arange(0.0,3.0,0.7)
print(creat_arr)
hba1c_arr = np.arange(0.0,20.0,4)
truefalse=[0,1]
id=0
list=[]
gender_list=["Male","Female"]

agegroup = np.arange(20,80,10)
for age in agegroup:
    for fbs in range(40,400,20):
        for pp2bs in range(40,400,20):
            for creatinine in creat_arr:
                #print(creatinine)
                for hba1c in hba1c_arr:
                    for urine_keto in truefalse:
                        for fundus in truefalse:
                            gender_Val = random.choice(gender_list)
                            temp=[id,gender_Val,age,fbs,pp2bs,creatinine,hba1c,urine_keto,fundus]
                            list.append(temp)
                            id=id+1

#print(list)
df = pd.DataFrame(list, columns = ['id','gender' ,'age','fbs','pp2bs','creatinine','hba1c','urine_ketoacidosis','fundus_retinotherapy'])

#df.to_csv('diabetes_data_low.csv',encoding='utf-8',index=False)


#df = pd.read_csv("diabetes_data_2.7.csv")

df["fbs_normal"] = np.random.randint(80,300,df.shape[0])

df["pp2bs_normal"] = np.random.randint(80,300,df.shape[0])


#df["age"]=np.random.randint(20,61,df.shape[0])

df["last_appointment"] = np.random.choice(pd.date_range('2019-09-01', '2020-01-21'), len(df))



#first_space=df.days.string.slice(,)

#for row in df['days']:
#    row = row.days
#    df['days']=row
   # print(row)
    
random_day = np.random.randint(low=5,high=90);
df['next_appointment'] = df["last_appointment"]+timedelta(days=random_day)


df['today_date'] = pd.to_datetime('today')

#df['days_from_next_appointment']=(df['today_date']-df['next_appointment']).dt.days
df['days_from_last_appointment']=(df['today_date']-df['last_appointment']).dt.days

df=df.drop(['today_date'],axis=1)
df = df.drop(['next_appointment'],axis=1)
df = df.drop(['last_appointment'],axis=1)
df['risk-factor'] =  df.apply(dia.calculate_risk1,axis=1)




df.to_csv("diabetes_data_2.7_with_risk.csv",encoding='utf-8',index=False)

df.to_excel("sampleformatdata.xlsx")
                
