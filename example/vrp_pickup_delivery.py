import pandas as pd
import numpy as np
from solomon import Solomon
import seaborn as sns
import matplotlib.pyplot as plt
import random
from copy import deepcopy


class Vrp(Solomon):

    def __init__(self):
        self.total_distance = 0
        self.routes = []
        self.routes_weight = []

    def read(self, file_name):
        # file_name = 'data/pickup&delivery.csv'
        df_vehlcle = pd.read_csv(file_name, skiprows=1, nrows=1)
        df_station = pd.read_csv(file_name, skiprows=5)
        self.data = df_station
        self.length = len(df_station) - 1
        self.vehicle = df_vehlcle
        self.capacity = df_vehlcle.capacity.iloc[0]
        self.unassign_station = list(range(1, self.length + 1))

    def update_routes(self):
        '''

        :return:
        '''
        while (len(self.unassign_station) > 0):
            # 从仓库出发选择一个站点和一辆车，使总距离最低
            best_cost = 10000
            is_new_route = 0
            best_station = -1
            best_route = -1
            best_position = -1

            for i in self.unassign_station:
                distance_cost = 2 * self.matrix[0, i]
                if distance_cost < best_cost:
                    best_cost = distance_cost
                    best_station = i
                    is_new_route = 1

                for r in range(len(self.routes)):
                    route = self.routes[r]
                    route_weight = self.routes_weight[r]
                    for k in range(1, len(route)):
                        if route_weight + self.data['weight'].iloc[i] > self.capacity:
                            continue
                        distance_cost = self.matrix[route[k - 1], i] + self.matrix[i, route[k]] - self.matrix[
                            route[k - 1], route[k]]

                        if distance_cost < best_cost:
                            best_cost = distance_cost
                            best_station = i
                            best_route = r
                            is_new_route = 0
                            best_position = k

            if is_new_route == 1:
                new_route = [0, best_station, 0]
                self.routes.append(new_route)
                self.routes_weight.append(self.data['weight'].iloc[best_station])
            else:
                route = self.routes[best_route]
                route.insert(best_position, best_station)
                self.routes_weight[best_route] = self.routes_weight[best_route] + self.data['weight'].iloc[best_station]
            self.total_distance = self.total_distance + best_cost
            self.unassign_station.remove(best_station)

    def cal_routes(self):
        # 初始化线路
        self.update_routes()
        print(self.total_distance)

        for i in range(10000):
            self.random_unassign_station(0.2)
            self.update_routes()
            print(i, self.total_distance)

    def random_unassign_station(self, rate):
        random_numbers = int(self.length * rate)
        unassign_station = random.sample(range(1, self.length + 1), random_numbers)
        for s in unassign_station:
            self.unassign_station.append(s)
            for i in range(len(self.routes)):
                route = self.routes[i]
                if s not in route:
                    continue
                else:
                    index = route.index(s)
                    distance_change = self.matrix[route[index - 1], route[index]] + self.matrix[
                        route[index], route[index + 1]] - self.matrix[route[index - 1], route[index + 1]]
                    self.routes_weight[i] = self.routes_weight[i] - self.data['weight'].iloc[s]
                    self.total_distance = self.total_distance - distance_change
                    route.remove(s)


    def plot(self):
        sns.relplot(x='x', y='y', data=self.data)
        plt.show()

    def plot_routes(self):
        points = self.data[['x', 'y']].values
        for path in self.routes:
            path = [int(i) for i in path]
            x = []
            y = []
            for p in path:
                x.append(points[p][0])
                y.append(points[p][1])
            sns.lineplot(x=x, y=y, sort=False)
        plt.show()


if __name__ == '__main__':
    vrp = Vrp()
    file_name = 'data/pickup&delivery.csv'
    vrp.read(file_name)
    vrp.compute_matrix()
    # vrp.cal_routes()
    vrp.update_routes()
    flag = 0
    for i in range(10000):
        if i - flag > 100:
            break
        vrp_copy = deepcopy(vrp)
        vrp_copy.random_unassign_station(0.2)
        vrp_copy.update_routes()

        if vrp_copy.total_distance < vrp.total_distance:
            vrp = vrp_copy
            flag = i
            print(i,vrp.total_distance)

    for i in vrp.routes:
        print(i)
    vrp.plot_routes()
