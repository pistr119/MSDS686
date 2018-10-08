# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 17:45:00 2018

@author: ipist
"""

#rnn

#part 1- data preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#only np arrays can be input arrays for keras
dataset_train = pd.read_csv('train.csv')
training_set = dataset_train #dataset_train.iloc[:, 1:2 ].values #left to right of columns, 2 here is excluded, only 1st index

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


#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled=sc.fit_transform(training_set)
 
#creating a data structure w/ 60 timesteps and 1 output.
X_train=[]
y_train=[]

#when there is a sigmoid function, best apply normalization rather than standardizatoin
for i in range(90,training_set.shape[0]): #i-60 to i.  Uppper bound is excluded ina range
    X_train.append(training_set_scaled[i-90:i,0]) #at each stock price, append prior 60 stock prices
    y_train.append(y_train_df[i]) #, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


#reshaping....
#Input shape:3D tensor with shape (batch_size, timesteps, input_dim). Timesteps=60 input_dim=could help predict.  
X_train=np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1) ) #new dimension, 3d-shape with indicator

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM 
from keras.layers import Dropout

regressor = Sequential() #predicting a continuous output, need to do regression.  
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))) #Long SHort Term Memory
#number of units - lstm cells/memory units in layer, return sequences = TRUE - stacked LSTM.  Last is input shape = 3D
regressor.add(Dropout(.2))#drop 20% of neurons.

#2nd layer
regressor.add(LSTM(units=50, return_sequences = True))
regressor.add(Dropout(0.2))

#3rd layer
regressor.add(LSTM(units=50, return_sequences = True))
regressor.add(Dropout(0.2))

#4th LSTM layer
regressor.add(LSTM(units=50, return_sequences= False))#False, no more return sequences. 
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1)) #units = num of neurons needs to be in output layer

#adam=always a safe choice, performs relevant updates of weights.  
regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=[ 'accuracy' ]) # loss=mean squared error.  Since this is a continuous value.  Predictions vs error.  
#I.e., close, other stock prices

regressor.fit(X_train, y_train, epochs=100, batch_size=32)