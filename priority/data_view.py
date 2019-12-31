import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/priority.csv')

sns.relplot(x='x',y = 'y',hue='priority', data = df, s=50, legend=False)

plt.show()