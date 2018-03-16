# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:58:04 2018

@author: Administrator
"""

import pandas as pd

def test_number(name):
    
    t0=pd.read_excel(name)
    
    jbs=pd.Series([1,2,3,4],index=['一线','二线','三线','四线'])
    wys=pd.Series([1,2,3],index=['普通住宅','别墅','商住'])
    
    t=t0.values
    
    
    for i in range(t.shape[0]):
        jb=t[i][3]
        t[i][3]=jbs[jb]
        wy=t[i][6]
        t[i][6]=wys[wy]
    t1=pd.DataFrame(t,columns=t0.columns)
    t1.to_excel('t1.xlsx')
    return t1

def train_number(name):
    
    t0=pd.read_excel(name)
    
    jbs=pd.Series([1,2,3,4],index=['一线','二线','三线','四线'])
    wys=pd.Series([1,2,3],index=['普通住宅','别墅','商住'])
    
    t=t0.values
    
    
    def put(fl,i,t):
        if fl==3:
            t[i][14]=1
        elif fl==5:
            t[i][16]=1
        elif fl==4:
            t[i][15]=1            
        elif fl==1:
            t[i][12]=1  
        elif fl==2:
            t[i][13]=1 
    
    
    for i in range(t.shape[0]):
        jb=t[i][3]
        t[i][3]=jbs[jb]
        wy=t[i][6]
        t[i][6]=wys[wy]
        fl=t[i][11]
        put(fl,i,t)
        
        
    t1=pd.DataFrame(t,columns=t0.columns)
    t1=t1.fillna(0)
    t1.to_excel('t1.xlsx')
    return t1



#train=train_number('train.xlsx')
#test=test_number('test.xlsx')

data_train = train.drop(['城市','区域','小区名'],axis=1) 


test_train = test.drop(['城市','区域','小区名'],axis=1) 


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
'''
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
'''

#dd=data_train.values
#tt=test_train.values

ct=pd.concat([data_train.iloc[:,0:8],test_train.iloc[:,0:8]])
data_max = ct.max()
data_min = ct.min()
data_train1 = (data_train.iloc[:,0:8]-data_min)/(data_max-data_min)  #数据标准化

y_train = data_train.iloc[:,9:].as_matrix() #训练样本标签列
x_train = data_train1.iloc[:,0:8].as_matrix() #训练样本特征

#y_test = data_test.iloc[:,4].as_matrix() #测试样本标签列


#pre = test.drop(['城市','区域','小区名'],axis=1)   

pre=test_train.iloc[:,0:8]            


pre1 = (pre-data_min)/(data_max-data_min)  #数据标准化

x_test=pre1.iloc[:,0:8].as_matrix() 

'''
x_train = dd[:,:8]
y_train =dd[:,9:]
x_test = dd[:,:8]
'''
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)


model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=8))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=5000,
          batch_size=128)
#score = model.evaluate(x_test, y_test, batch_size=128)

r=model.predict(x_test)
print(r)
re=pd.DataFrame(r)
re.to_excel('re.xls')

y_predict=[]

def rou(k):
    return round(k,3)

for i in range(r.shape[0]):
    f=list(r[i])
    f1=list(map(rou,f))
    dek=f1.index(max(f1))
    y_predict.append(dek+1)
    
    
test['客户分类']=y_predict 

test.to_excel('test_out4.xls')   




'''
data_train = train.drop(['城市','区域','小区名'],axis=1) 

data_max = data_train.max()
data_min = data_train.min()
data_train1 = (data_train-data_min)/(data_max-data_min)  #数据标准化

y_train = data_train1.iloc[:,7:8].as_matrix() #训练样本标签列
x_train = data_train1.iloc[:,0:7].as_matrix() #训练样本特征

#y_test = data_test.iloc[:,4].as_matrix() #测试样本标签列

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

model = Sequential() #建立模型
model.add(Dense(input_dim = 7, output_dim = 240)) #添加输入层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数
model.add(Dense(input_dim = 240, output_dim = 120)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数
model.add(Dense(input_dim = 120, output_dim = 120)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数
model.add(Dense(input_dim = 120, output_dim = 1)) #添加隐藏层、输出层的连接
model.add(Activation('sigmoid')) #以sigmoid函数为激活函数
#编译模型，损失函数为binary_crossentropy，用adam法求解
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, nb_epoch = 100, batch_size = 800) #训练模型
model.save_weights('net.model') #保存模型参数



#inputfile2='gg.xls' #预测数据
pre = test.drop(['城市','区域','小区名'],axis=1)                


pre1 = (pre-pre.min())/(pre.max()-pre.min())  #数据标准化

pre2 = pre1.iloc[:,0:7].as_matrix() #预测样本特征                 
r = pd.DataFrame(model.predict(pre2))


rt=r*(pre.max()-pre.min())()+pre.min()
print(r.round(2))
'''    
  