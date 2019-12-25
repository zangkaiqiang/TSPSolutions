import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

class solomon:
    def __init__(self):
        self.data = pd.DataFrame()
        self.vehicle = []

    def read_solomons(self, filename):
        '''
        read solomon format data
        :param filename:
        :return: Dataframe, list
        '''
        data = np.loadtxt(filename, skiprows=9)
        # CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME
        self.data = pd.DataFrame(data, columns=['id', 'x', 'y', 'demand', 'ready_time', 'due_data', 'service_time'])
        self.vehicle = np.loadtxt(filename, skiprows=4, max_rows=1)
        self.nums = len(self.data) - 1
        self.points = self.data[['x', 'y']].values

    def compute_matrix(self):
        '''
        # 计算距离矩阵
        :param df:
        :return:
        '''
        size = len(self.data)
        matrix = np.ndarray((size, size))
        points = self.data[['x', 'y']].values
        for i in range(size):
            for j in range(size):
                matrix[i][j] = self.compute_distance(points[i], points[j])
        self.matrix = matrix

    @staticmethod
    def compute_distance(p1, p2):
        '''

        :param p1:
        :param p2:
        :return:
        '''
        return np.math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def eval_path(self, path):
        '''
        评估path的长度
        :param path:
        :return:
        '''
        length = self.matrix[0][path[0]]
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            length = length + self.matrix[p1][p2]
        length = length + self.matrix[path[-1]][0]
        return length

    def eval_paths(self, paths):
        dis_list = []
        for path in paths:
            dis_list.append(self.eval_path(path))
        best_distance = min(dis_list)
        best_path = paths[dis_list.index(best_distance)]
        return best_distance, best_path

    def find_neighbors(self, path):
        length = len(path)
        neighbors = []
        for i in range(1, length - 1):
            for j in range(i + 1, length):
                path_copy = path.copy()
                path_copy[i], path_copy[j] = path_copy[j], path_copy[i]
                neighbors.append(path_copy)
        return neighbors

    def tabu_search(self):
        size = len(self.data)
        paths = []
        init_path = random.sample(range(1, size), size-1)
        paths.append(init_path)
        tabu_table = []
        table_len = 20

        tabu_table.append(init_path)

        best_distance, best_path = self.eval_paths(paths)
        expect_distace, expect_path = best_distance, best_path

        for i in range(5000):
            neighbors = self.find_neighbors(best_path)
            best_distance,best_path = self.eval_paths(neighbors)

            if best_distance < expect_distace:
                expect_distace = best_distance
                expect_path = best_path
                if best_path in tabu_table:
                    tabu_table.remove(best_path)
                    tabu_table.append(best_path)
                else:
                    tabu_table.append(best_path)
            else:
                if best_path in tabu_table:
                    neighbors.remove(best_path)
                    ##
                    best_distance, best_path = self.eval_paths(neighbors)
                    tabu_table.append(best_path)
                else:
                    tabu_table.append(best_path)

            if len(tabu_table) >= table_len:
                del tabu_table[0]
        return expect_path, expect_distace

    def plot(self,path):
        x = []
        y = []
        for p in path:
            x.append(self.points[p][0])
            y.append(self.points[p][1])
        # 创建图并命名
        plt.figure('Line fig')
        ax = plt.gca()
        # 设置x轴、y轴名称
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
        # 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度
        ax.plot(x, y, color='r', linewidth=1, alpha=0.6)

        plt.show()


