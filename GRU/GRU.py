import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense, Activation ,LeakyReLU
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from alpha_vantage.timeseries import TimeSeries

# #google sheets
# from __future__ import print_function
# from apiclient.discovery import build
# from httplib2 import Http
# from oauth2client import file, client, tools

# def get_google_sheet(spreadsheet_id, range_name):
#     """ Retrieve sheet data using OAuth credentials and Google Python API. """
#     scopes = 'https://www.googleapis.com/auth/spreadsheets.readonly'
#     # Setup the Sheets API
#     store = file.Storage('credentials.json')
#     creds = store.get()
#     if not creds or creds.invalid:
#         flow = client.flow_from_clientsecrets('client_secret.json', scopes)
#         creds = tools.run_flow(flow, store)
#     service = build('sheets', 'v4', http=creds.authorize(Http()))

#     # Call the Sheets API
#     gsheet = service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
#     return gsheet

# def gsheet2df(gsheet):
#     """ Converts Google sheet data to a Pandas DataFrame.
#     Note: This script assumes that your data contains a header file on the first row!
#     Also note that the Google API returns 'none' from empty cells - in order for the code
#     below to work, you'll need to make sure your sheet doesn't contain empty cells,
#     or update the code to account for such instances.
#     """
#     header = gsheet.get('values', [])[0]   # Assumes first line is header!
#     values = gsheet.get('values', [])[1:]  # Everything else is data.
#     if not values:
#         print('No data found.')
#     else:
#         all_data = []
#         for col_id, col_name in enumerate(header):
#             column_data = [] 
#             for row in values:
#                 column_data.append(row[col_id])
#             ds = pd.Series(data=column_data, name=col_name)
#             all_data.append(ds)
#         df = pd.concat(all_data, axis=1)
#         return df

def featureScaling(training_set):
  sc = MinMaxScaler(feature_range=(0,1))
  training_set_scaled = sc.fit_transform(training_set)
  return training_set_scaled

def train(train_x, train_y):
  train_x, train_y = np.array(train_x), np.array(train_y)
  train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
  
  model = Sequential()

  model.add(GRU(units=50,return_sequences=True,input_shape=(train_x.shape[1], 1)))
  model.add(Dropout(0.2))
  model.add(GRU(units=50,return_sequences=True))
  model.add(Dropout(0.2))
  model.add(GRU(units=50,return_sequences=True))
  model.add(Dropout(0.2))
  model.add(GRU(units=50))
  model.add(Dropout(0.2))
  model.add(Dense(10))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Dropout(0.2))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))

  model.compile(optimizer='adam',loss='mean_squared_error')

  print("......GO TO SLEEP................")
  model.fit(train_x,train_y,epochs=20,batch_size=64)
  print("......GET UP!!!!!!!!!!!!!!!")

  model.save('GRU.h5')
  return model

def PlotGraph(predict_y, predicted_stock_price):
  plt.plot(predict_y, color = 'yellow', label = 'Original Stock Price')
  plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Stock Price')
  plt.title('APOLLOTYRE__EQ__NSE__NSE__MINUTE Stock Price Prediction')
  plt.xlabel('Time')
  plt.ylabel('APOLLOTYRE__EQ__NSE__NSE__MINUTE Stock Price')
  plt.legend()
  plt.show()

# !pip install alpha_vantage
def last100values(company):
  company_symbol = 'NSE:'+ company
  ts = TimeSeries(key='XAYE00R4HWQOB5SU', output_format='pandas')
  data, meta_data = ts.get_intraday(symbol=company_symbol,interval='5min', outputsize='combat')
  training_set = data.iloc[:, 1:2].values
  return training_set
  # print(company_symbol)

# values = last100values('infy')
# PlotGraph(values, values)

def predict_value( x , model):
  predict_x = np.array(x)
  predict_x = np.reshape(predict_x, (predict_x.shape[0], predict_x.shape[1], 1))
  predicted_y = model.predict(predict_x)
  return predicted_y

company = 'APOLLOTYRE'
model = load_model('../WeightFile/GRU.h5')

while True:
  #to get the last 100 values
  predict_x = last100values(company)

  #feature scaling
  sc = MinMaxScaler(feature_range=(0,1))
  x_scaled = sc.fit_transform(predict_x)

  #predicting the next value
  predicted_x = predict_value(x_scaled, model)

  #reverse feature scaling
  predicted_stock_price = sc.inverse_transform(predicted_x)

  #RMSE value
  print("Final rmse value is =",np.sqrt(np.mean((predicted_x - predicted_stock_price)**2)))

  #plot the results
  PlotGraph(predict_x, predicted_stock_price)

  # predicting in every 1 minute
  time.sleep(1)