from core.vrp import vrp


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
        return True


    def check_weight(self,route):
        '''

        :param route:
        :return:
        '''
        weight = 0
        for i in route:
            weight = weight+ self.weight[i]
            if weight>self.capacity:
                return False
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