import pandas as pd
file_name = 'data/pd-100.csv'

df_vehlcle = pd.read_csv(file_name, skiprows=1, nrows=1)
df_station = pd.read_csv(file_name, skiprows=5)