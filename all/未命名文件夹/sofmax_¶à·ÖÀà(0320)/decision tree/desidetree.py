#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 10:51:12 2017

@author: luogan
"""
#read data
import pandas as pd
df = pd.read_csv('loans.csv')

#print(df.head())

X = df.drop('safe_loans', axis=1)


y = df.safe_loans


#change categorical

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
d = defaultdict(LabelEncoder)
X_trans = X.apply(lambda x: d[x.name].fit_transform(x))
X_trans.head()

#X_trans.to_excel('X_trans.xls') 

#random take train and test

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_trans, y, random_state=1)
#call decision tree
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=8)
clf = clf.fit(X_train, y_train)





test_rec = X_test.iloc[1,:]
clf.predict([test_rec])

y_test.iloc[1]
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, clf.predict(X_test)))
















