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
training_set = dataset_train.iloc[:, 1:4 ].values #left to right of columns 


#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1), copy=True)

sales_scaled=sc.fit_transform(training_set[:,2:3])
training_set_scaled=training_set[:,0:2]
sales_scaled_vector=sales_scaled[:,0]
training_set_scaled=np.hstack((training_set_scaled, sales_scaled ))
#training_set_scaled[2]=sales_scaled_vector[0]
#training_set_scaled=sc.fit_transform(training_set[3])

#creating a data structure w/ 90 timesteps and 1 output.
X_train=[]
y_train=[]

upper=len(dataset_train)
lower=90 #predict 3 months
#when there is a sigmoid function, best apply normalization rather than standardizatoin
for i in range(lower,upper): #i-90 to i.  Uppper bound is excluded ina range
    X_train.append(training_set_scaled[i-lower:i,0]) #at each sale price, append prior 90 same amounts
    y_train.append(training_set_scaled[i, 0])
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
regressor.compile(optimizer='adam', loss='mean_squared_error') # loss=mean squared error.  Since this is a continuous value.  Predictions vs error.  
#I.e., close, other stock prices

regressor.fit(X_train, y_train, epochs=100, batch_size=32)
 

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2 ].values #left to right of columns, 2 here is excluded, only 1st index


dataset_total=pd.concat((dataset_train['Open'], dataset_test['Open']),axis=0) #concat train and test datasets #vertical concat - axis=0
#Part 3 - making predicitons and visualizing results.  
inputs= dataset_total[len(dataset_total)-len(dataset_test) - 60:].values #.values converts to numpy array
inputs = inputs.reshape(-1, 1)

#3d format expected by neural network for training and predictions
inputs = sc.transform(inputs)

X_test=[]

for i in range(60, 80): #60 + 20 (1 month financial days)
    X_test.append(inputs[i-60:i, 0])
X_test=np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) #reverse normalization


plt.plot(real_stock_price, color = 'red' , label='Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue' , label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



