# import keras
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np
# import warnings
# import time
# warnings.filterwarnings("ignore")
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Dense, Activation ,LeakyReLU
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import load_model
# from alpha_vantage.timeseries import TimeSeries

# # #google sheets
# # from __future__ import print_function
# # from apiclient.discovery import build
# # from httplib2 import Http
# # from oauth2client import file, client, tools

# # def get_google_sheet(spreadsheet_id, range_name):
# #     """ Retrieve sheet data using OAuth credentials and Google Python API. """
# #     scopes = 'https://www.googleapis.com/auth/spreadsheets.readonly'
# #     # Setup the Sheets API
# #     store = file.Storage('credentials.json')
# #     creds = store.get()
# #     if not creds or creds.invalid:
# #         flow = client.flow_from_clientsecrets('client_secret.json', scopes)
# #         creds = tools.run_flow(flow, store)
# #     service = build('sheets', 'v4', http=creds.authorize(Http()))

# #     # Call the Sheets API
# #     gsheet = service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
# #     return gsheet

# # def gsheet2df(gsheet):
# #     """ Converts Google sheet data to a Pandas DataFrame.
# #     Note: This script assumes that your data contains a header file on the first row!
# #     Also note that the Google API returns 'none' from empty cells - in order for the code
# #     below to work, you'll need to make sure your sheet doesn't contain empty cells,
# #     or update the code to account for such instances.
# #     """
# #     header = gsheet.get('values', [])[0]   # Assumes first line is header!
# #     values = gsheet.get('values', [])[1:]  # Everything else is data.
# #     if not values:
# #         print('No data found.')
# #     else:
# #         all_data = []
# #         for col_id, col_name in enumerate(header):
# #             column_data = [] 
# #             for row in values:
# #                 column_data.append(row[col_id])
# #             ds = pd.Series(data=column_data, name=col_name)
# #             all_data.append(ds)
# #         df = pd.concat(all_data, axis=1)
# #         return df

# def featureScaling(training_set):
#   sc = MinMaxScaler(feature_range=(0,1))
#   training_set_scaled = sc.fit_transform(training_set)
#   return training_set_scaled

# def train(train_x, train_y):
#   train_x, train_y = np.array(train_x), np.array(train_y)
#   train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
  
#   model = Sequential()

#   model.add(LSTM(units=50,return_sequences=True,input_shape=(train_x.shape[1], 1)))
#   model.add(Dropout(0.2))
#   model.add(LSTM(units=50,return_sequences=True))
#   model.add(Dropout(0.2))
#   model.add(LSTM(units=50,return_sequences=True))
#   model.add(Dropout(0.2))
#   model.add(LSTM(units=50))
#   model.add(Dropout(0.2))
#   model.add(Dense(10))
#   model.add(LeakyReLU(alpha=0.1))
#   model.add(Dropout(0.2))
#   model.add(Dense(1))
#   model.add(Activation('sigmoid'))

#   model.compile(optimizer='adam',loss='mean_squared_error')

#   print("......GO TO SLEEP................")
#   model.fit(train_x,train_y,epochs=20,batch_size=64)
#   print("......GET UP!!!!!!!!!!!!!!!")

#   model.save('LSTM.h5')
#   return model

# def PlotGraph(predict_y, predicted_stock_price):
#   plt.plot(predict_y, color = 'yellow', label = 'Original Stock Price')
#   plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Stock Price')
#   plt.title('APOLLOTYRE__EQ__NSE__NSE__MINUTE Stock Price Prediction')
#   plt.xlabel('Time')
#   plt.ylabel('APOLLOTYRE__EQ__NSE__NSE__MINUTE Stock Price')
#   plt.legend()
#   plt.show()

# # !pip install alpha_vantage
# def last100values(company):
#   company_symbol = 'NSE:'+ company
#   ts = TimeSeries(key='XAYE00R4HWQOB5SU', output_format='pandas')
#   data, meta_data = ts.get_intraday(symbol=company_symbol,interval='5min', outputsize='combat')
#   training_set = data.iloc[:, 1:2].values
#   return training_set
#   # print(company_symbol)

# # values = last100values('infy')
# # PlotGraph(values, values)

# def predict_value( x , model):
#   predict_x = np.array(x)
#   predict_x = np.reshape(predict_x, (predict_x.shape[0], predict_x.shape[1], 1))
#   predicted_y = model.predict(predict_x)
#   return predicted_y

# company = 'APOLLOTYRE'
# model = load_model('../WeightFile/LSTM.h5')

# while True:
#   #to get the last 100 values
#   predict_x = last100values(company)

#   #feature scaling
#   sc = MinMaxScaler(feature_range=(0,1))
#   x_scaled = sc.fit_transform(predict_x)

#   #predicting the next value
#   predicted_x = predict_value(x_scaled, model)

#   #reverse feature scaling
#   predicted_stock_price = sc.inverse_transform(predicted_x)

#   #RMSE value
#   print("Final rmse value is =",np.sqrt(np.mean((predicted_x - predicted_stock_price)**2)))

#   #plot the results
#   PlotGraph(predict_x, predicted_stock_price)

#   # predicting in every 1 minute
#   time.sleep(1)


import os
import time
import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Conv1D,MaxPooling1D,GlobalAveragePooling1D,GRU
from tensorflow.keras.layers import Dropout,Flatten,RepeatVector
from tensorflow.keras.layers import Dense, Activation ,LeakyReLU
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
# from alpha_vantage.timeseries import TimeSeries

def featureScaling(training_set):
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)
    return training_set_scaled,sc

def train(train_x, train_y):
    train_x, train_y = np.array(train_x), np.array(train_y)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
  

    model = Sequential()

    model.add(LSTM(units=64,return_sequences=True,input_shape=(train_x.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer='adam',loss='mean_squared_error')


    print("......GO TO SLEEP................")
    model.fit(train_x,train_y,epochs=50,batch_size=32)
    print("......GET UP!!!!!!!!!!!!!!!")

    model.save('cnngru.h5')
    return model

def retrain(train_x, train_y):
    train_x, train_y = np.array(train_x), np.array(train_y)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))


    model = load_model('weight_file.h5')
    print("......GO TO SLEEP................")
    model.fit(train_x,train_y,epochs=30,batch_size=64)
    print("......GET UP!!!!!!!!!!!!!!!")
    model.save("new.h5")
    return model


def PlotGraph(predict_y, predicted_stock_price):
    plt.plot(predict_y, color = 'yellow', label = 'Original Stock Price')
    plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Stock Price')
    plt.title('APOLLOTYRE__EQ__NSE__NSE__MINUTE Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('APOLLOTYRE__EQ__NSE__NSE__MINUTE Stock Price')
    plt.legend()
    plt.show()


def predict_value(predict_x, model):
    predict_x = np.array(predict_x)
    predict_x = np.reshape(predict_x, (predict_x.shape[0], predict_x.shape[1], 1))
    predicted_y = model.predict(predict_x)
    return predicted_y


dataset_train = pd.read_csv("/content/INFY__EQ__NSE__NSE__15__MINUTE.csv")
dataset_train["timestamp"] = pd.to_datetime(dataset_train["timestamp"])
dataset_train.dropna(axis = 0, how ='any',inplace=True)

training_set = dataset_train.iloc[:, 2:3].values

training_set_scaled,sc = featureScaling(training_set)

partition = int(len(training_set_scaled)*0.1)
start_of_test_set = len(training_set_scaled)- partition

train_x =[]
train_y =[]
for i in range(40, len(training_set_scaled)-partition):
    train_x.append(training_set_scaled[i-40:i, 0])
    train_y.append(training_set_scaled[i, 0])


predict_x =[]
predict_y =[]
for i in range(start_of_test_set, len(training_set_scaled)):
    predict_x.append(training_set_scaled[i-40:i, 0])
    predict_y.append(training_set_scaled[i, 0])


model = train(train_x, train_y)

predict_x, predict_y = np.array(predict_x), np.array(predict_y)
predict_x = np.reshape(predict_x, (predict_x.shape[0], predict_x.shape[1], 1))

predicted_y = model.predict(predict_x)
predicted_stock_price = sc.inverse_transform(predicted_y)
predict_y=predict_y.reshape(-1,1)
predict_y = sc.inverse_transform(predict_y)

PlotGraph(predict_y, predicted_stock_price)

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(predict_y, predicted_stock_price))
print(f"RMSE is : {rmse}")

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(predict_y,predicted_stock_price)
print(f"MAE is : {mae}")
