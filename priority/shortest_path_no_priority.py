import pandas as pd
import numpy as np
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/priority.csv')

length = len(df)
points = df[['x', 'y']].values
start_porint = points[0]
priority_list = df.loc[df['priority'] == 1, 'id'].tolist()
start_time = df['start_time'].tolist()
end_time = df['end_time'].tolist()
service_time = df['service_time'].tolist()
priority = df['priority'].tolist()


def cal_matrix(points):
    length = len(points)
    distance_matrix = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            distance_matrix[i, j] = cal_distance(points[i], points[j])

    return distance_matrix


def cal_distance(p1, p2):
    return np.math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def path_length(path):
    total_distance = 0
    for i, j in zip(path[:-1], path[1:]):
        total_distance = total_distance + mat[i, j]
    return total_distance


def check_path(path):
    for i, j in zip(path[:-1], path[1:]):
        if start_time[i] + service_time[i] > end_time[j]:
            return False
    return True


def plot_path(path):
    x = []
    y = []
    for p in path:
        x.append(points[p][0])
        y.append(points[p][1])
    sns.scatterplot(x=x, y=y, style=priority, legend=False, s=100)
    # sns.lineplot(x=x, y=y, sort=False)
    plt.plot(x, y)
    plt.savefig('output/shortest_path.png')
    plt.show()


def priority_path():
    pass

if __name__ == '__main__':
    mat = cal_matrix(points)
    unassign = list(range(length))

    path = [0]
    for i in path:
        unassign.remove(i)
    while len(unassign) > 0:
        min_distance = 1 << 100
        best = 0
        index = 0

        path_copy = deepcopy(path)
        origin_distance = path_length(path_copy)
        for i in unassign:
            for j in range(len(path_copy)):
                path_copy.insert(j + 1, i)
                # if not check_path(path_copy):
                #     continue
                new_distance = path_length(path_copy)
                if new_distance < min_distance:
                    best = i
                    min_distance = new_distance
                    index = j
                path_copy.remove(i)

        path.insert(index + 1, best)
        unassign.remove(best)
    print(path)
    plot_path(path)



