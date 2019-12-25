from core.vrppd import vrppd
from copy import deepcopy

def pd_solution():
    vrp = vrppd()
    file_name = 'data/pd-100.csv'
    vrp.read_pd(file_name)
    vrp.compute_matrix()
    # vrp.cal_routes()
    vrp.update_routes()
    flag = 0
    for i in range(10000):
        if i - flag > 20:
            break
        vrp_copy = deepcopy(vrp)
        vrp_copy.random_unassign_station(0.2)
        vrp_copy.update_routes()

        if vrp_copy.total_distance < vrp.total_distance:
            vrp = vrp_copy
            flag = i
            print(i, vrp.total_distance)

    for i in vrp.routes:
        print(i)
    vrp.plot_routes()


if __name__ == '__main__':
    pd_solution()