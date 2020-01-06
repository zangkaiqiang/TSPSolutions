from core.vrp import vrp
from copy import deepcopy


class vrptw(vrp):
    def __init__(self):
        super().__init__()

    def check(self, route):
        '''

        :param route:
        :return:
        '''
        if self.check_weight(route) is False:
            return False
        if self.check_tw(route) is False:
            return False
        return True

    def check_weight(self, route):
        '''
        载重检测
        :param route:
        :return:
        '''
        weight = 0
        for i in route:
            weight = weight + self.weight[i]
            if weight > self.capacity:
                return False
        return True

    def check_tw(self, route):
        '''
        时间窗检测
        :param route:
        :return:
        '''
        first = route[1]
        last_finish_time = self.ready_time[first] + self.service_time[first]
        for i in route[2:-1]:
            if last_finish_time > self.due_time[i]:
                return False
            ready_time = max(self.ready_time[i], last_finish_time)
            last_finish_time = ready_time + self.service_time[i]
        return True

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

                    for k in range(1, len(route)):
                        route_copy = deepcopy(route)
                        route_copy.insert(k, i)
                        if self.check(route_copy) is False:
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

def solomon_solution():
    vrp_case = vrptw()
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
        if len(i)>2:
            print(i)
    vrp_case.plot_routes('solomon-tw')

if __name__ == '__main__':
    solomon_solution()