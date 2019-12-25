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