import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

points_num = 100
car_num = 20

def generate():
    np.random.seed(0)
    # id,x,y,weight,type,capacity, service_time, start_time, end_time

    x = np.random.randint(1, 1000, points_num)
    y = np.random.randint(1, 1000, points_num)
    weight = np.random.randint(10, 50, points_num)
    x[0] = 500
    y[0] = 500

    type = np.zeros(points_num)
    capacity = np.zeros(points_num)
    # 仓库
    type[0] = 1
    # 车假设有9辆车
    type[1:car_num] = 2
    capacity[1:car_num] = np.random.choice((200,300,400),car_num-1)

    service_time = np.random.randint(1,100, points_num)
    start_time = np.zeros(points_num)
    end_time= np.zeros(points_num) + 2000

    service_time[0:car_num] = 0
    weight[0:car_num] = 0

    df = pd.DataFrame({'x':x,'y':y})
    df['weight'] = weight
    df['type'] = type
    df['capacity'] = capacity
    df['service_time'] = service_time
    df['start_time'] = start_time
    df['end_time'] = end_time

    df.to_csv('data/withgoods.csv',index_label='id')

def view():
    df = pd.read_csv('data/withgoods.csv')
    palette = sns.color_palette('muted', n_colors=3)
    # ax = sns.scatterplot(x='x',y='y',hue='type',data=df, s=50, legend=False,palette= palette)
    df['size'] = df['weight'] + df['capacity']

    ax = sns.scatterplot(x='x', y='y', data = df,hue='type', size='size',palette=palette,)
    # sns.scatterplot(x='x', y='y', data = df,hue='type', size='capacity',palette=palette, ax=ax)
    # sns.scatterplot(x='x', y='y', data = df[df['type'] == 2], size='capacity', ax=ax,legend='full')
    title = 'return with goods - location'
    # plt.title(title)
    ax.set_title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig('output/%s.png'%title)
    plt.show()

if __name__ == '__main__':
    generate()
    view()
