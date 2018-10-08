# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 13:25:58 2018

@author: ipist
"""
#part 1- data preprocessing
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

def build_series(df, series_size=3):
    for i in range(series_size):
        for column in df.columns:
            df['%s%s' % (column , i)] = df[column].shift(i)
            
            
#dimension columns
def prepare_data(df, series_size=3):
    #col_list = ['store', 'item', 'day', 'month', 'year', 'weekday', 'weekend', 'monthbegin', 'monthend', 'quarter') #), 'filled_series', 'rank']
    col_list = ['store', 'item', 'day', 'month', 'year', 'weekday', 'weekend', 'monthbegin', 'monthend', 'quarter']
    res_list = []
    for row in range(df.shape[0]): #shape[0]=rows, shape[1]=cols
        row_array=[]
        for col in col_list:
            for time in range(series_size):
                row_array.append(df[col + '%s' % time].values)
            res_list.append(np.asarray(row_array))
    return np.asarray(res_list)

dataset_train = pd.read_csv('train.csv')
y_train_df= dataset_train['sales'].values #.values
 
training_set = dataset_train #dataset_train.iloc[:, 0:4 ].values #left to right of columns 
#training_set = training_set.drop('sales', axis=1)
training_set['date']=pd.to_datetime(training_set['date'])
training_set['day']=pd.DatetimeIndex(training_set['date']).day
training_set['month']=pd.DatetimeIndex(training_set['date']).month
training_set['year']=pd.DatetimeIndex(training_set['date']).year
training_set['weekday']=(pd.DatetimeIndex(training_set['date']).weekday < 5) #  #sat-sun
training_set['weekend']=(pd.DatetimeIndex(training_set['date']).weekday >= 5) #(0,1,2,3,4) #mon-fri
training_set['monthbegin']=pd.DatetimeIndex(training_set['date']).is_month_start
training_set['monthend']=pd.DatetimeIndex(training_set['date']).is_month_end
training_set['quarter']=pd.DatetimeIndex(training_set['date']).quarter
training_set=training_set.drop('date', axis=1)
 
training_set=training_set.drop('sales', axis=1)
 
build_series(training_set, series_size=2)

#prepare_data(training_set, series_size=1)