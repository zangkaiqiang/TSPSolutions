import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/priority.csv')

sns.relplot(x='x', y='y', hue='priority', data=df, s=200, legend=False)
plt.savefig('output/priority-source.png')
plt.show()
