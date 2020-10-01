import pandas as pd
import os
from pathlib import Path

path = Path(os.getcwd())
csv_file_path = path.parent

dataset = pd.read_csv(csv_file_path+"/INFY__EQ__NSE__NSE__MINUTE.csv")
dataset = dataset.iloc[:,0:2]
indices=[]
for i in range(len(dataset)):
    if i%15 != 0:
        indices.append(i)
     
new_dataset = dataset.drop(indices)
new_dataset = new_dataset.reset_index()
new_dataset = new_dataset.drop(['index'], axis=1)
new_dataset.to_csv(csv_file_path+"/INFY__EQ__NSE__NSE__15__MINUTE.csv")
os.remove(csv_file_path+"/INFY__EQ__NSE__NSE__MINUTE.csv")