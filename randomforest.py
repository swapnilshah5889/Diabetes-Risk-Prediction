import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import diabetes as dia      
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
def truefalse(x):
    x = x.strip()
    if x == 'true' or x == 'True' or x == True:
        return 1
    else:
        return 0

dataset = pd.read_csv('diabetes_data_2.7_with_risk.csv')

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




#applying k fold crossvalidagtion accuracy portion

#from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(estimator = classifier,X=X_train,y=y_train,cv=10)
#print(accuracies.mean())