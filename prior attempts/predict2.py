# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:32:08 2018

@author: ipist
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
 
 
from keras.layers import LSTM 

def build_series(df, series_size=3):
    for i in range(series_size):
        for column in df.columns:
            df['%s%s' % (column , i)] = df[column].shift(i)
            
            
#dimension columns
def prepare_data(df, series_size=3):
    col_list = ['store', 'item', 'day', 'month', 'year', 'weekday', 'weekend', 'monthbegin', 'monthend', 'yearquarter', 'filled_series', 'rank']
    res_list = []
    for row in range(df.shape[0]): #shape[0]=rows, shape[1]=cols
        row_array=[]
        for col in col_list:
            for time in range(series_size):
                row_array.append(df[col + '%s' % time].values)
            res_list.append(np.asarray(row_array))
    return np.asarray(res_list)
    

dataset_train = pd.read_csv('train.csv')
dataset_train = dataset_train.drop(['date'], axis=1)
dataset_test = pd.read_csv('test.csv')
dataset_test = dataset_test.drop(['date'],  axis=1)

y_train= dataset_train['sales'].values #values converts to np array
type(y_train)

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1), copy=True)


sc=MinMaxScaler()
sc.fit(dataset_train)

dataset_train=dataset_train.dropna()
#sales_scaled=sc.fit_transform(dataset_train[:,2:3])
sales_scaled=sc.fit_transform(dataset_train[:,'sales'])
training_set_scaled=dataset_train[:,0:2]
sales_scaled_vector=sales_scaled[:,0]
training_set_scaled=np.hstack((training_set_scaled, sales_scaled ))

test_id_column=dataset_test['id']
dataset_test=dataset_test.drop(['id'], axis=1) #no longer need it

build_series(dataset_train)



