#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 21:14:08 2017

@author: luogan
"""

#read data
import pandas as pd
df = pd.read_csv('loans.csv')

#print(df.head())

#X = df.drop('safe_loans', axis=1)

X = df.drop(['safe_loans' ],axis=1)
y = df.safe_loans


#change categorical

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
d = defaultdict(LabelEncoder)
X_trans = X.apply(lambda x: d[x.name].fit_transform(x))
X_trans.head()

#X_trans.to_excel('X_trans.xls') 
##############
data_train=X_trans
data_max = data_train.max()
data_min = data_train.min()
data_mean = data_train.mean()
#
# data_std = data_train.std()
X_train1 = (data_train-data_max)/(data_max-data_min)


y=0.5*(y+1)
#random take train and test

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_train1, y, random_state=1)


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

model = Sequential() #建立模型
model.add(Dense(input_dim = 12, output_dim = 48)) #添加输入层、隐藏层的连接
model.add(Activation('tanh')) #以Relu函数为激活函数

model.add(Dense(input_dim = 48, output_dim = 48)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数
model.add(Dropout(0.2))

model.add(Dense(input_dim = 48, output_dim = 36)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数
model.add(Dropout(0.2))
model.add(Dense(input_dim = 36, output_dim = 36)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数

model.add(Dense(input_dim = 36, output_dim = 12)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数
model.add(Dense(input_dim = 12, output_dim = 12)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数


model.add(Dense(input_dim = 12, output_dim = 1)) #添加隐藏层、输出层的连接
model.add(Activation('sigmoid')) #以sigmoid函数为激活函数
#编译模型，损失函数为binary_crossentropy，用adam法求解
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train.values, y_train.values, nb_epoch = 70, batch_size = 2000) #训练模型

r = pd.DataFrame(model.predict_classes(x_test.values))
'''
r = pd.DataFrame(model.predict(x_test.values))
rr=r.values
tr=rr.flatten()

for i in range(tr.shape[0]):
    if tr[i]>0.5:
        tr[i]=1
    else:
        
        tr[i]=0
''' 

print(/n)         
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, r))        