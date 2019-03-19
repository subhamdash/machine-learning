# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 23:22:52 2019

@author: subham
"""

import pandas as pd
from sklearn import svm
import xgboost

from sklearn.linear_model import LinearRegression


training_set =pd.read_excel("Data_Train.xlsx")
test_set = pd.read_excel("Test_set.xlsx")
test_set1 = pd.read_excel("Test_set.xlsx")
training_set.head()
training_set = training_set.dropna()
test_set=test_set.dropna()


training_set['Date_of_Journey']=pd.to_datetime(training_set.Date_of_Journey)
training_set['Journey_Day'] = pd.to_datetime(training_set.Date_of_Journey, format='%d/%m/%Y').dt.day
training_set['Journey_Month'] = pd.to_datetime(training_set.Date_of_Journey, format='%d/%m/%Y').dt.month
training_set.drop(labels = 'Date_of_Journey', axis = 1, inplace = True)

training_set['Arrival_Time']=pd.to_datetime(training_set.Arrival_Time)
training_set['Arr_Time_Hour'] = pd.to_datetime(training_set.Arrival_Time).dt.hour
training_set['Arr_Time_Minutes'] = pd.to_datetime(training_set.Arrival_Time).dt.minute
training_set.drop(labels = 'Arrival_Time', axis = 1, inplace = True)

training_set['Dep_Time']=pd.to_datetime(training_set.Dep_Time)
training_set['Depart_Time_Hour'] = pd.to_datetime(training_set.Dep_Time).dt.hour
training_set['Depart_Time_Minutes'] = pd.to_datetime(training_set.Dep_Time).dt.minute
training_set.drop(labels = 'Dep_Time', axis = 1, inplace = True)




test_set['Date_of_Journey']=pd.to_datetime(test_set.Date_of_Journey)
test_set['Journey_Day']=pd.to_datetime(test_set.Date_of_Journey,format='%d%m%Y').dt.day
test_set['Journey_Month']=pd.to_datetime(test_set.Date_of_Journey,format='%d%m%Y').dt.month
test_set.drop(labels='Date_of_Journey',axis=1,inplace=True)

test_set['Arrival_Time']=pd.to_datetime(test_set.Arrival_Time)
test_set['Arr_Time_Hour'] = pd.to_datetime(test_set.Arrival_Time).dt.hour
test_set['Arr_Time_Minutes'] = pd.to_datetime(test_set.Arrival_Time).dt.minute
test_set.drop(labels = 'Arrival_Time', axis = 1, inplace = True)

test_set['Dep_Time']=pd.to_datetime(test_set.Dep_Time)
test_set['Depart_Time_Hour'] = pd.to_datetime(test_set.Dep_Time).dt.hour
test_set['Depart_Time_Minutes'] = pd.to_datetime(test_set.Dep_Time).dt.minute
test_set.drop(labels = 'Dep_Time', axis = 1, inplace = True)


duration = list(training_set['Duration'])

for i in range(len(duration)) :
    if len(duration[i].split()) != 2:
        if 'h' in duration[i] :
            duration[i] = duration[i].strip() + ' 0m'
        elif 'm' in duration[i] :
            duration[i] = '0h {}'.format(duration[i].strip())

dur_hours = []
dur_minutes = []  

for i in range(len(duration)) :
    dur_hours.append(int(duration[i].split()[0][:-1]))
    dur_minutes.append(int(duration[i].split()[1][:-1]))
    
training_set['Duration_hours'] = dur_hours
training_set['Duration_minutes'] =dur_minutes







durationT = list(test_set['Duration'])

for i in range(len(durationT)) :
    if len(durationT[i].split()) != 2:
        if 'h' in durationT[i] :
            durationT[i] = durationT[i].strip() + ' 0m'
        elif 'm' in durationT[i] :
            durationT[i] = '0h {}'.format(durationT[i].strip())
            
dur_hours = []
dur_minutes = []  

for i in range(len(durationT)) :
    dur_hours.append(int(durationT[i].split()[0][:-1]))
    dur_minutes.append(int(durationT[i].split()[1][:-1]))
  
    
test_set['Duration_hours'] = dur_hours
test_set['Duration_minutes'] = dur_minutes


Y_train = training_set.iloc[:,7].values  
X_train = training_set.iloc[:,training_set.columns != 'Price']


test_set.drop(labels = 'Duration', axis = 1, inplace = True)
X_train.drop(labels = 'Duration', axis = 1, inplace = True)
X_test = test_set

"test_set = test_set.iloc[:,:].values"

"X_test=pd.DataFrame(X_test)"
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
le2 = LabelEncoder()
for col in X_train.columns.values:

       if X_train[col].dtypes=='object':
           data=X_train[col].append(X_train[col])
           le1.fit_transform(data.values)
           X_train[col]=le1.fit_transform(X_train[col])
         
for col in X_test.columns.values:

       if X_test[col].dtypes=='object':
           data=X_test[col].append(X_test[col])
           le2.fit_transform(data.values)
           X_test[col]=le2.fit_transform(X_test[col])
           

   





from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
test_set = sc_X.fit_transform(test_set)
Y_train = Y_train.reshape((len(Y_train), 1)) 
Y_train = sc_X.fit_transform(Y_train)
Y_train = Y_train.ravel()





clf = xgboost.XGBRegressor()
clf.fit(X_train,Y_train)

model=LinearRegression()
model.fit(X_train,Y_train)


yu=model.predict(test_set)

Y_pred=clf.predict(test_set)
Y_pred=sc_X.inverse_transform(Y_pred)


























from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_train)# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 15:43:55 2019

@author: subham
"""




