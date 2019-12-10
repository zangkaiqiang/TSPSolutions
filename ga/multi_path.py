# 多路径优化
import random
import math
import numpy as np

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

# 假设前两个点为target
n = 20

np.random.seed(0)
x = np.random.random(n) * 100
y = np.random.random(n) * 100
# 距离矩阵，计算每两个点的距离
dis = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        pi = (x[i], y[i])
        pj = (x[j], y[j])
        dij = math.sqrt((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2)
        dis[i, j] = dij

# 生成每个点的target
target = np.random.random(n) * 2
target = [int(i) for i in target]
# 先不考虑出发点


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.sample, list(range(n-1)), n-1)
# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.attr_bool)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_p10(individual):
    point_d = {}
    length = dis[0][individual[0]+1]
    point_d[individual[0]+1] = length
    for i in range(len(individual) - 1):
        length = length+dis[individual[i]+1][individual[i+1]+1]
        point_d[individual[i+1]+1] = length
    # 假设第十个点到第0个点的距离有限制不能太远
    return length + point_d[10],

def eval(individual):
    length = dis[0][individual[0]+1]
    for i in range(len(individual) - 1):
        length = length+dis[individual[i]+1][individual[i+1]+1]
    return length,


toolbox.register("evaluate", eval)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)


def short_path():
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1000,
                                   stats=stats, halloffame=hof, verbose=True)

    fitness_value = np.array([i.fitness.values[0] for i in pop])
    path = pop[fitness_value.argmin()]
    path = [i+1 for i in path]

    return path,log

if __name__ == '__main__':
    path, log = short_path()
    print(path)
    # print(eval(np.array(path)-1))

#[9, 3, 7, 8, 1, 2, 5, 6, 4]
#[3, 11, 5, 1, 18, 7, 19, 17, 13, 8, 10, 2, 12, 9, 6, 4, 14, 16, 15]