%matplotlib inline
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import os
import pandas as pd 
from pathlib import Path


ts = TimeSeries(key='3IV6H0ADFCU3X5UQ', output_format='pandas')
newdata, newmeta_data = ts.get_intraday(symbol='AAPL',interval='1min', outputsize='full')

path = Path(os.getcwd())
csv_file_path = path.parent


data = pd.read_csv(csv_file_path+"/AAPL.csv")

data = pd.merge(data,newdata,how='outer')

data.to_csv(csv_file_path+"/AAPL.csv")
