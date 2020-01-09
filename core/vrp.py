import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from copy import deepcopy

class vrp():

    def __init__(self):
        self.total_distance = 0
        self.routes = []
        self.routes_weight = []
        self.data = None
        self.vehicle = None
        self.capacity = 0
        self.length = 0
        self.unassign_station = []
        self.weight = []
        self.points = []


    def read_solomon(self, filename):
        '''

        :param filename:
        :return:
        '''
        data = np.loadtxt(filename, skiprows=9)
        # CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME
        self.data = pd.DataFrame(data, columns=['id', 'x', 'y', 'weight', 'ready_time', 'due_time', 'service_time'])
        vehicle = np.loadtxt(filename, skiprows=4, max_rows=1)
        # df_vehicle = pd.DataFrame(vehicle,columns=['number','capacity'])
        self.capacity = vehicle[1]


        self.length = len(self.data) - 1
        self.unassign_station = list(range(1, self.length + 1))

        self.pickup = np.zeros(len(self.data))
        self.weight = self.data['weight'].tolist()
        self.points = self.data[['x', 'y']].values
        self.ready_time = self.data['ready_time'].tolist()
        self.due_time = self.data['due_time'].tolist()
        self.service_time = self.data['service_time'].tolist()


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
                        if route_weight + self.weight[i] > self.capacity:
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

    def plot(self,title='source'):
        sns.relplot(x='x', y='y', data=self.data)
        plt.savefig('output/%s.png'%title)
        plt.show()

    def plot_routes(self,title='output'):
        for path in self.routes:
            path = [int(i) for i in path]
            x = []
            y = []
            point_type = []
            for p in path:
                x.append(self.points[p][0])
                y.append(self.points[p][1])
                point_type.append(self.pickup[p])
            sns.scatterplot(x=x,y=y,style=point_type,legend=False,s=100)
            # sns.lineplot(x=x, y=y, sort=False)
            plt.plot(x, y)
        plt.savefig('output/%s.png'%title)
        plt.show()

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


def solomon_solution():
    vrp_case = vrp()
    file_name = 'data/solomon-100/In/r101.txt'
    vrp_case.read_solomon(file_name)
    vrp_case.compute_matrix()
    # vrp.cal_routes()
    vrp_case.update_routes()
    flag = 0
    for i in range(10000):
        if i - flag > 100:
            break
        vrp_copy = deepcopy(vrp_case)
        vrp_copy.random_unassign_station(0.2)
        vrp_copy.update_routes()

        if int(vrp_copy.total_distance * 1000) < int(vrp_case.total_distance * 1000):
            vrp_case = vrp_copy
            flag = i
            print(i, vrp_case.total_distance)

    for i in vrp_case.routes:
        print(i)
    vrp_case.plot_routes()


def pd_solution():
    vrp_case = vrp()
    file_name = 'data/pd-100.csv'
    vrp_case.read_pd(file_name)
    vrp_case.compute_matrix()
    # vrp.cal_routes()
    vrp_case.update_routes()
    flag = 0
    for i in range(10000):
        if i - flag > 20:
            break
        vrp_copy = deepcopy(vrp_case)
        vrp_copy.random_unassign_station(0.2)
        vrp_copy.update_routes()

        if vrp_copy.total_distance < vrp_case.total_distance:
            vrp_case = vrp_copy
            flag = i
            print(i, vrp_case.total_distance)

    for i in vrp_case.routes:
        print(i)
    vrp_case.plot_routes('solomon-without-tw')

if __name__ == '__main__':
    solomon_solution()
