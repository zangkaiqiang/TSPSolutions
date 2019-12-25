from core.vrp import vrp
from copy import deepcopy

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
    vrp_case.plot_routes()

if __name__ == '__main__':
    pd_solution()