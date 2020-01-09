import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from copy import deepcopy
from core.vrp import vrp

points_num = 100
car_num = 20


class vrpreturn(vrp):
    def __init__(self):
        super().__init__()

    def load_data(self, fp):
        df = pd.read_csv(fp)
        self.data = df
        self.capacity = df['capacity']

        self.length = points_num - car_num
        self.unassign_station = list(range(car_num, points_num))

        self.vehicles = list(range(1,car_num))

        self.weight = self.data['weight'].tolist()
        self.points = self.data[['x', 'y']].values
        self.ready_time = self.data['start_time'].tolist()
        self.due_time = self.data['end_time'].tolist()
        self.service_time = self.data['service_time'].tolist()

    def init_routes(self):
        for v in self.vehicles:
            self.routes.append([v,0])
            self.routes_weight.append(0)

    def update_routes(self):
        while (len(self.unassign_station) > 0):
            best_cost = 1<<100
            best_station = -1
            best_route = -1
            best_position = -1
            for i in self.unassign_station:
                for r in range(len(self.routes)):
                    route = self.routes[r]
                    route_weight = self.routes_weight[r]
                    capacity = self.capacity[route[0]]

                    for k in range(1, len(route)):
                        # 超载
                        if route_weight + self.weight[i] > capacity:
                            continue

                        distance_cost = self.matrix[route[k - 1], i] + self.matrix[i, route[k]] - self.matrix[
                            route[k - 1], route[k]]
                        if distance_cost < best_cost:
                            best_cost = distance_cost
                            best_station = i
                            best_route = r
                            best_position = k

            route = self.routes[best_route]
            route.insert(best_position, best_station)
            self.routes_weight[best_route] = self.routes_weight[best_route] + self.data['weight'].iloc[best_station]

            self.total_distance = self.total_distance + best_cost
            if best_station not in self.unassign_station:
                print('Overload!')
            self.unassign_station.remove(best_station)

    def random_unassign_station(self, rate):
        random_numbers = int(self.length * rate)
        unassign_station = np.random.choice(list(range(car_num, points_num)), random_numbers)
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

    def solution(self, max_iter = 10000):
        self.init_routes()
        self.update_routes()
        flag = 0
        copy1 = deepcopy(self)
        for i in range(max_iter):
            if i - flag > 100:
                break
            vrp_copy = deepcopy(copy1)
            vrp_copy.random_unassign_station(0.2)
            vrp_copy.update_routes()

            if int(vrp_copy.total_distance * 1000) < int(copy1.total_distance * 1000):
                copy1 = vrp_copy
                flag = i
                print(i, copy1.total_distance)

        for i in copy1.routes:
            print(i)
        self.plot_routes('return with goods')

    def plot_routes(self, title):
        ax = sns.scatterplot(x='x',y='y', data=self.data[self.data['type']==0], size='weight')
        sns.scatterplot(x='x',y='y', data= self.data[self.data['type'] ==2],size='capacity',ax=ax,legend='full')
        for path in self.routes:
            path = [int(i) for i in path]
            x = []
            y = []
            for p in path:
                x.append(self.points[p][0])
                y.append(self.points[p][1])
            plt.plot(x,y)
        ax.set_title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.savefig('output/%s.png'%title)
        plt.show()


if __name__ == '__main__':
    fp = 'data/withgoods.csv'
    vrp_case = vrpreturn()
    vrp_case.load_data(fp)
    vrp_case.compute_matrix()
    vrp_case.solution()