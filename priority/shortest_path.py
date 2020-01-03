import pandas as pd
import numpy as np
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt

class Path:
    def __init__(self):
        self.points = []
        self.path = []

    def read_csv(self, filename):
        df = pd.read_csv(filename)
        self.length = len(df)
        self.points = df[['x', 'y']].values
        self.start_point = self.points[0]
        self.start_time = df['start_time'].tolist()
        self.end_time = df['end_time'].tolist()
        self.service_time = df['service_time'].tolist()
        self.priority = df['priority'].tolist()
        self.priority_list = df.loc[df['priority'] == 1, 'id'].tolist()


    def build(self):
        self.cal_matrix()

    def cal_matrix(self):
        length = len(self.points)
        distance_matrix = np.zeros((length, length))
        for i in range(length):
            for j in range(length):
                distance_matrix[i, j] = self.cal_distance(self.points[i], self.points[j])

        self.mat = distance_matrix

    def cal_distance(self, p1, p2):
        return np.math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def path_length(self, path):
        total_distance = 0
        for i, j in zip(path[:-1], path[1:]):
            total_distance = total_distance + self.mat[i, j]
        return total_distance

    def check_path(self, path):
        ac_start_time = [0]
        for i in range(1, len(path)):
            ast = max( self.service_time[path[i - 1]] + ac_start_time[i - 1],  self.start_time[path[i]])
            if ast >  self.end_time[path[i]]:
                return False
            ac_start_time.append(ast)
        return True

    def plot(self):
        x = []
        y = []
        style = []
        for p in self.path:
            x.append(self.points[p][0])
            y.append(self.points[p][1])
            style.append(self.priority[p])
        sns.scatterplot(x=x, y=y, hue=style, legend=False, s=200)
        plt.plot(x, y)
        plt.savefig('output/priority.png')
        plt.show()

    def get_path(self):
        unassign = list(range(self.length))
        priority_list = deepcopy(self.priority_list)
        path = [0]
        while (len(priority_list) > 0):
            min_distance = 1 << 100
            best = 0
            index = 0
            if len(path) == 1:
                for i in priority_list:
                    distance = self.mat[0, i]
                    if distance < min_distance:
                        best = i
                        min_distance = distance
                        index = 0
            else:
                path_copy = deepcopy(path)
                for i in priority_list:
                    for j in range(len(path_copy)):
                        path_copy.insert(j + 1, i)
                        if not self.check_path(path_copy):
                            path_copy.remove(i)
                            continue
                        new_distance = self.path_length(path_copy)
                        if new_distance < min_distance:
                            best = i
                            min_distance = new_distance
                            index = j
                        path_copy.remove(i)
            path.insert(index + 1, best)
            priority_list.remove(best)

        for i in path:
            unassign.remove(i)

        while len(unassign) > 0:
            min_distance = 1 << 100
            best = 0
            index = 0

            path_copy = deepcopy(path)
            for i in unassign:
                for j in range(len(path_copy)):
                    path_copy.insert(j + 1, i)
                    if not self.check_path(path_copy):
                        path_copy.remove(i)
                        continue
                    new_distance = self.path_length(path_copy)
                    if new_distance < min_distance:
                        best = i
                        min_distance = new_distance
                        index = j
                    path_copy.remove(i)

            path.insert(index + 1, best)
            unassign.remove(best)

        self.path = path

    def get_path2(self):
        unassign = list(range(self.length))
        path = [0]
        for i in path:
            unassign.remove(i)
        while len(unassign) > 0:
            min_distance = 1 << 100
            best = 0
            index = 0

            path_copy = deepcopy(path)
            for i in unassign:
                for j in range(len(path_copy)):
                    path_copy.insert(j + 1, i)
                    if not self.check_path(path_copy):
                        path_copy.remove(i)
                        continue
                    new_distance = self.path_length(path_copy)
                    if new_distance < min_distance:
                        best = i
                        min_distance = new_distance
                        index = j
                    path_copy.remove(i)

            path.insert(index + 1, best)
            unassign.remove(best)

        self.path = path

    def info(self):
        print(self.path)
        print(self.path_length(self.path))


if __name__ == '__main__':
    path = Path()
    path.read_csv('data/priority.csv')
    path.build()
    path.get_path()
    path.plot()
    path.info()

