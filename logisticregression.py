# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:02:53 2020

@author: hardik
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("final_diabetes.csv",index_col=0)

df = dataset.copy()

df=df.drop(['address','email','mobile-number'],axis=1)

print(df)

df1 =df.copy()