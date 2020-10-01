import os
import time
import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense, Activation ,LeakyReLU
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from alpha_vantage.timeseries import TimeSeries


def last100values(company):
  company_symbol = 'NSE:'+ company
  ts = TimeSeries(key='XAYE00R4HWQOB5SU', output_format='pandas')
  data, meta_data = ts.get_intraday(symbol=company_symbol,interval='15min', outputsize='compact')
  training_set = data.iloc[:, 1:2].values
  return training_set


def featureScaling(training_set):
  sc = MinMaxScaler(feature_range=(0,1))
  training_set_scaled = sc.fit_transform(training_set)
  return training_set_scaled,sc


def predict_value(predict_x, model):
  predict_x = np.array(predict_x)
  predict_x = np.reshape(predict_x, (predict_x.shape[0], predict_x.shape[1], 1))
  predicted_y = model.predict(predict_x)
  return predicted_y


company = 'INFY'
model = load_model('weight_file.h5')

while True:
  #to get the last 100 values
    last100 = last100values(company)#last 100
    predict_x= last100.iloc[0:60,:]#latest 60


    #feature scaling
    sc = MinMaxScaler(feature_range=(0,1))
    x_scaled = sc.fit_transform(predict_x)

    #predicting the next value
    predicted_x = predict_value(x_scaled, model)

    #reverse feature scaling
    predicted_stock_price = sc.inverse_transform(predicted_x)

    #plot the results
    #PlotGraph(predict_x, predicted_stock_price)

    print(predicted_stock_price)

    # predicting in every 1 minute
    time.sleep(15)


