import pandas as pd
import numpy as np

np.random.seed(1)
points_num = 21

x = np.random.randint(1,1000,points_num)
y = np.random.randint(1,1000,points_num)
x[0] = 500
y[0] = 500
df = pd.DataFrame({'x':x,'y':y})

service_time = np.random.randint(1,100, points_num)
service_time[0] = 0
start_time = np.zeros(points_num)
end_time= np.zeros(points_num) + 2000
priority_list = np.zeros(points_num)

priority = np.random.choice(range(points_num), 2, replace=False)
for i in range(len(priority)):
    p = priority[i]
    start_time[p] = 200*(i+1)
    end_time[p] = start_time[p] + 50
    priority_list[p] = 1

df['priority'] = priority_list
df['service_time'] = service_time
df['start_time'] = start_time
df['end_time'] = end_time

df.to_csv('data/priority.csv', index_label='id')


