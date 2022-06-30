# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 18:57:52 2022

@author: User
"""
import pandas as pd
#import seaborn as sns
import ta
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
# import autokeras as ak







crypto_data = pd.read_csv(r'D:\kaggle_project\g-research-crypto-forecasting\supplemental_train.csv', index_col=('Asset_ID'), encoding='utf8', engine='python').dropna()

crypto_data['timestamp'] = pd.to_datetime(crypto_data['timestamp'], origin='unix', unit='s')#, )
#crypto_data.set_index('Asset_ID')
crypto_data.sort_index(inplace=True)

crypto_data['Returns'] = crypto_data.Close.pct_change()
crypto_data['Log_Returns'] = np.log(1+crypto_data['Returns'])


crypto_data['RSI'] = ta.momentum.RSIIndicator(crypto_data['Close']).rsi()
crypto_data.dropna(inplace=True)
X = crypto_data[['Close', 'Log_Returns', 'RSI']].values
scaler = MinMaxScaler(feature_range = (0,1)).fit(X)
X_scaled = scaler.transform(X)
y = [x[0] for x in X_scaled]
split = int(len(X_scaled)*0.8)
X_train = X_scaled[:split]
X_test = X_scaled[split:len(X_scaled)]
y_train = y[:split]
y_test = y[split:len(y)]
#print(len(X_train) == len(y_train))
#print(len(X_test) == len(y_test))
n = 90
Xtrain = []
Xtest = []
ytrain = []
ytest = []

for i in range(n, len(X_train)):
    Xtrain.append(X_train[i-n:i, :X_train.shape[1]])
    ytrain.append(y_train[i])
for i in range(n, len(X_test)):
    Xtest.append(X_test[i-n:i, :X_test.shape[1]])
    ytest.append(y_test[i])

Xtrain , ytrain = np.array(Xtrain) , np.array(ytrain)
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2]))

Xtest , ytest = np.array(Xtest) , np.array(ytest)
Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], Xtest.shape[2]))

#print(Xtrain.shape)
#print(ytrain.shape)
#print(Xtest.shape)
#print(ytest.shape)

predict_from = 1
predict_until = 10
lookback = 3
# clf = ak.TimeseriesForecaster(
#     lookback=lookback,
#     predict_from=predict_from,
#     predict_until=predict_until,
#     max_trials=1,
#     objective="val_loss",
# )
    

model = ak.Sequential(lookback=lookback,
    predict_from=predict_from,
    predict_until=predict_until,
    max_trials=1,
    objective="val_loss",)
model.add(LSTM(input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
model.add(Dropout())
model.add(Dense())
model.compile(loss='mse', optimizer='adam')
path = 'D:\kaggle_project'
checkpoint = ModelCheckpoint(filepath=path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
callbacks =[checkpoint, earlystopping]
history = model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), verbose=1, callbacks=callbacks)
plt.Figure(figsize=(16, 7))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
trainpredict = model.predict(Xtrain)
testpredict = model.predict(Xtest)
#trainpredict = np.c_[trainpredict, np.zeros(trainpredict.shape)]
#testpredict = np.c_[testpredict, np.zeros(testpredict.shape)]
trainpredict = scaler.inverse_transform(trainpredict)#.reshape(-1, 1)
trainpredict = [x[0] for x in trainpredict]
testpredict = scaler.inverse_transform(testpredict)#.reshape(-1, 1)
testpredict = [x[0] for x in testpredict]

print(trainpredict[:5])
print(testpredict[:5])

#print(crypto_data.head()) 
#i = 0  
#for i in range(14):
#    cryptoi = crypto_data[crypto_data.index == i]
#    plt.plot(cryptoi['timestamp'], cryptoi['Close'])
#    plt.plot(cryptoi['timestamp'], cryptoi['RSI'])

#    plt.show()
#print(crypto_data.tail(20))
#sns.scatterplot(x=crypto_data['timestamp'], y=crypto_data['Close'], data=crypto_data, hue=crypto_data['Asset_ID']) 
#df = crypto_data.groupby(by='Asset_ID')
#df.sort_values(by=['Asset_ID'], axis=0, ascending=True)
#print(df.head())
