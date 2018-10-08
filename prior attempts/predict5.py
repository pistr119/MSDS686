import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler


def build_series(df, serie_size=3):
    for i in range(serie_size):
        for column in df.columns:
            df['%s%s' % (column, i)] = df[column].shift(i)


def prepare_data(df, serie_size=3):
    #columns = ['store', 'item', 'day', 'month', 'year', 'weekday', 'weekend', 'monthbegin', 'monthend', 'yearquarter', 'filled_serie', 'rank']
    columns = ['store', 'item', 'day', 'month', 'year', 'weekday', 'weekend', 'monthbegin', 'monthend', 'quarter']
    list_result = []
    for row in range(df.shape[0]):
        row_array = []
        for column in columns:
            for time in range(serie_size):
                row_array.append(df[column+'%s' % time].values)
        list_result.append(np.asarray(row_array))
    return np.asarray(list_result)


train = pd.read_csv('train.csv') #.drop(['date' ], axis=1)
#validation = pd.read_csv('data/db/validation.csv').drop(['date', 'dateFormated'], axis=1)
#test = pd.read_csv('data/db/test.csv').drop(['date', 'dateFormated'], axis=1)

train_y = train['sales'].values
#validation_y = validation['sales'].values
#test_ids = test['id']
#test = test.drop(['id'], axis=1)
#train = train.drop(['sales'], axis=1)
#validation = validation.drop(['sales'], axis=1)

training_set = train #dataset_train.iloc[:, 0:4 ].values #left to right of columns 
training_set = training_set.drop('sales', axis=1)
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

#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler(feature_range=(0,1), copy=True)
#training_set_scaled=sc.fit_transform(training_set)
        
X_train=[]
y_train=[]

np_training_set=np.array(training_set)

for i in range(90,training_set.shape[0]): #i-60 to i.  Uppper bound is excluded ina range
    X_train.append(np_training_set[i-90:i,0]) #at each stock price, append prior 60 stock prices
    y_train.append(train_y[i ] ) #, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
 