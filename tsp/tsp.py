import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy


class tsp:
    def load_data(self, fp):
        df = pd.read_csv(fp)
        self.data = df
        self.points = self.data[['x', 'y']].values
        self.length = len(self.data)

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
        path_length = 0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            path_length = path_length + self.matrix[p1][p2]
        return path_length

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
        paths = []
        init_path = self.path_init()
        paths.append(init_path)
        tabu_table = []
        table_len = 100

        tabu_table.append(init_path)

        best_distance, best_path = self.eval_paths(paths)
        expect_distace, expect_path = best_distance, best_path

        for i in range(5000):
            neighbors = self.find_neighbors(best_path)
            best_distance, best_path = self.eval_paths(neighbors)

            print(best_distance)

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

    def plot(self, path):
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

    def path_update(self, unassign):
        path = deepcopy(self.path)
        for i in unassign:
            if i in path:
                path.remove(i)

        while len(unassign) > 0:
            min_distance = 1 << 100
            best = 0
            index = 0

            path_copy = deepcopy(path)
            for i in unassign:
                for j in range(len(path_copy)):
                    path_copy.insert(j + 1, i)

                    new_distance = self.eval_path(path_copy)
                    if new_distance < min_distance:
                        best = i
                        min_distance = new_distance
                        index = j
                    path_copy.remove(i)

            path.insert(index + 1, best)
            unassign.remove(best)

        return path

    def path_init(self):
        unassign = list(range(1, len(self.data)))
        path = [0]
        self.path = path
        self.path = self.path_update(unassign)
        print(self.eval_path(self.path))
        return self.path

    def local_search(self):
        self.load_data('data/tsp.csv')
        self.compute_matrix()

        self.path_init()
        best_length = self.eval_path(self.path)
        for i in range(1000):
            unassign = list(np.random.choice(range(1, self.length), int(self.length * 0.2)))
            new_path = self.path_update(unassign)
            new_length = self.eval_path(new_path)
            if new_length<best_length:
                self.path = new_path
                best_length = new_length
                print(new_length)


    def tabu_run(self):
        self.load_data('data/tsp.csv')
        self.compute_matrix()
        self.tabu_search()


if __name__ == '__main__':
    tsp_case = tsp()
    tsp_case.local_search()
    tsp_case.plot(tsp_case.path)
