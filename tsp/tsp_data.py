import pandas as pd
import numpy as np

np.random.seed(0)
point_num = 200

x = np.random.randint(1,1000,point_num)
y = np.random.randint(1,1000,point_num)
x[0] = 500
y[0] = 500
df = pd.DataFrame({'x':x,'y':y})

df.to_csv('data/tsp.csv',index_label='id')