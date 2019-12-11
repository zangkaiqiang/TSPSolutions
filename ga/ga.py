# 遗传算法解决TSP
import random
import math
import numpy as np
import pandas as pd

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

np.random.seed(1)

df_data = pd.read_csv('data/data.csv')
df_vehicle = pd.read_csv('data/vehicle.csv')
n = len(df_data)
x = df_data['x']
y = df_data['y']

# 距离矩阵，计算每两个点的距离
dis = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        pi = (x[i], y[i])
        pj = (x[j], y[j])
        dij = math.sqrt((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2)
        dis[i, j] = dij

# # 生成每个点的target
# target = np.random.random(n) * 2
# target = [int(i) for i in target]
# 先不考虑出发点


creator.create("FitnessMin", base.Fitness, weights=(-1.0, -0.1, 1))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.sample, list(range(n)), n)
# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.attr_bool)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 使用s2型车 120
w = 120
def eval(individual, df_data):
    distance = 0
    total_distance = 0
    loading_rate = []
    total_weight = 0
    length = len(individual)
    for i in range(length):
        weight = df_data.loc[individual[i], 'weight']
        total_weight = total_weight + weight
        if total_weight > w:
            total_distance = total_distance + distance
            distance = 0
            loading_rate.append((total_weight - weight) / w)
            total_weight = weight
            continue
        if i == length - 1:
            loading_rate.append(total_weight / w)
            total_distance = total_distance + distance
        else:
            distance = distance + dis[individual[i]][individual[i + 1]]

    loading_rate = np.array(loading_rate)
    avg_loading_rate = loading_rate.mean()
    vehicle_num = len(loading_rate)

    select = (total_distance * vehicle_num) / avg_loading_rate
    return vehicle_num, total_distance, avg_loading_rate

def get_path(individual):
    total_weight = 0
    length = len(individual)
    start = 0
    path = []
    for i in range(length):
        weight = df_data.loc[individual[i], 'weight']
        total_weight = total_weight + weight
        if total_weight > w:
            if i == start:
                raise RuntimeError('%d too big'%individual[i])
            total_weight = weight
            path.append(individual[start:i])
            start = i
    return path

toolbox.register("evaluate", eval, df_data=df_data)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
# toolbox.register("select", tools.selNSGA2)
toolbox.register("select", tools.selSPEA2)


def ga():
    pop = toolbox.population(n=400)
    hof = tools.HallOfFame(3)
    stats1 = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats2 = tools.Statistics(lambda ind: ind.fitness.values[1])
    stats3 = tools.Statistics(lambda ind: ind.fitness.values[2])
    mstats = tools.MultiStatistics(distance=stats1, num=stats2, rate=stats3)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.5, ngen=1,
                                   stats=mstats, halloffame=hof, verbose=True)

    return hof, log


if __name__ == '__main__':
    hof, log = ga()
    print(hof.keys)
    item = hof.items[0]
    path = get_path(item)
    df_path = pd.DataFrame({'item':item})

