# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:53:57 2020

@author: hardik
"""


import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
import diabetes as dia
df = pd.read_csv("diabetes_data_2.7.csv")

df["fbs_normal"] = np.random.randint(80,300,df.shape[0])

df["pp2bs_normal"] = np.random.randint(80,300,df.shape[0])


df["age"]=np.random.randint(20,61,df.shape[0])

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

df['risk-factor'] =  df.apply(dia.calculate_risk1,axis=1)




df.to_csv("diabetes_data_2.7_with_risk.csv",encoding='utf-8',index=False)


