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


creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,1))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.sample, list(range(n)), n)
# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.attr_bool)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# 使用s2型车 120
w = 1000
def eval(individual):
    distance = 0
    total_distance = 0
    loading_rate = []
    weight = 0
    length = len(individual)
    for i in range(length):
        weight = weight + df_data.loc[individual[i], 'weight']
        if weight > w:
            total_distance = total_distance + distance
            distance = 0
            weight = df_data.loc[individual[i], 'weight']
            loading_rate.append(weight / w)
            continue
        if i == length - 1:
            loading_rate.append(weight / w)
            total_distance = total_distance + distance
        else:
            distance = distance + dis[individual[i]][individual[i + 1]]

    loading_rate = np.array(loading_rate)
    avg_loading_rate = loading_rate.mean()
    vehicle_num = len(loading_rate)

    select = (total_distance * vehicle_num) / avg_loading_rate
    return select, vehicle_num, avg_loading_rate


toolbox.register("evaluate", eval)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


def ga():
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(3)
    stats_distance = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_num = tools.Statistics(lambda ind: ind.fitness.values[1])
    stats_rate = tools.Statistics(lambda ind: ind.fitness.values[2])
    mstats = tools.MultiStatistics(distance=stats_distance,num=stats_num,rate=stats_rate)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.5, ngen=100,
                                   stats=mstats, halloffame=hof, verbose=True)

    # fitness_value = np.array([i.fitness.values[0] for i in pop])

    return hof, log


if __name__ == '__main__':
    hof, log = ga()
    print(hof.keys)
