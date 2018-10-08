# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 13:58:47 2018

@author: ipist
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('train.csv')
y_train= dataset_train['sales'].values
training_set = dataset_train.iloc[:, 1:4 ].values #left to right of columns 


#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1), copy=True)

sales_scaled=sc.fit_transform(training_set[:,2:3])
training_set_scaled=training_set[:,0:2]
#sales_scaled_vector=sales_scaled[:,0]
training_set_scaled=np.hstack((training_set_scaled, sales_scaled ))

train_cols=['store', 'item', 'sales']
train=pd.DataFrame(training_set_scaled)
train.columns =train_cols
y_train_scaled=train['sales'].values
train = train.drop(['sales'], axis=1)

