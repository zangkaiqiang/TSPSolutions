import random

import pandas as pd
import numpy as np

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import seaborn as sns
import matplotlib.pyplot as plt


def read_solomon(filename):
    '''
    read solomon format data
    :param filename:
    :return: Dataframe, list
    '''
    data = np.loadtxt(filename, skiprows=9)
    # CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME
    df = pd.DataFrame(data, columns=['id', 'x', 'y', 'demand', 'ready_time', 'due_data', 'service_time'])
    vehicle = np.loadtxt(filename, skiprows=4, max_rows=1)
    return df, vehicle


def compute_matrix(df):
    '''
    # 计算距离矩阵
    :param df:
    :return:
    '''
    size = len(df)
    matrix = np.ndarray((size, size))
    points = df[['x', 'y']].values
    for i in range(size):
        for j in range(size):
            matrix[i][j] = compute_distance(points[i], points[j])

    return matrix


def compute_distance(p1, p2):
    '''

    :param p1:
    :param p2:
    :return:
    '''
    return np.math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def eval(individual, distance_matrix, w, df_data):
    '''

    :param individual:
    :param distance_matrix:
    :param w:
    :param df_data:
    :return:
    '''
    # 重置个体在距离矩阵中的index
    individual = [i+1 for i in individual]

    distance = 0
    total_distance = 0
    loading_rate = []
    total_weight = 0
    length = len(individual)
    for i in range(length):
        # 初始距离为起始仓库到第一个站点的距离
        if distance == 0:
            distance = distance_matrix[0][individual[i]]
        else:
            distance = distance + distance_matrix[individual[i - 1]][individual[i]]

        weight = df_data.loc[individual[i], 'demand']
        total_weight = total_weight + weight

        # 当整体载重大于了车辆载重
        if total_weight > w:
            total_distance = total_distance + distance + distance_matrix[0][individual[i - 1]] - \
                             distance_matrix[individual[i - 1]][individual[i]]
            distance = distance_matrix[0][individual[i]]
            loading_rate.append((total_weight - weight) / w)
            total_weight = weight
            continue

        # 处理最后一个节点
        if i == length - 1:
            loading_rate.append(total_weight / w)
            total_distance = total_distance + distance + distance_matrix[0][individual[i]]
            loading_rate.append(total_weight / w)

    loading_rate = np.array(loading_rate)
    avg_loading_rate = loading_rate.mean()
    vehicle_num = len(loading_rate)

    # select = (total_distance * vehicle_num) / avg_loading_rate
    return total_distance, vehicle_num


def ga():
    df, vehicle = read_solomon('data/solomon-100/In/c101.txt')
    matrix = compute_matrix(df)
    size = len(df) - 1

    creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("attr_bool", random.sample, list(range(size)), size)
    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_bool)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval, distance_matrix=matrix, df_data=df, w=vehicle[1])
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)
    # toolbox.register("select", tools.selSPEA2)

    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(3)
    stats1 = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats2 = tools.Statistics(lambda ind: ind.fitness.values[1])
    # stats3 = tools.Statistics(lambda ind: ind.fitness.values[2])
    mstats = tools.MultiStatistics(distance=stats1, num=stats2)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1000,
                                   stats=mstats, halloffame=hof, verbose=True)

    path = get_path(hof.items[0], df, vehicle[1])

    return hof, log, path


def get_path(individual,df,w):
    '''

    :param individual:
    :param df:
    :param w:
    :return:
    '''
    individual = [i+1 for i in individual]

    total_weight = 0
    length = len(individual)
    start = 0
    path = []
    for i in range(length):
        weight = df.loc[individual[i], 'demand']
        total_weight = total_weight + weight
        if total_weight > w:
            if i == start:
                raise RuntimeError('%d too big'%individual[i])
            total_weight = weight
            path.append(individual[start:i])
            start = i
    return path

def plot_route():
    df, vehicle = read_solomon('data/solomon-100/In/c101.txt')
    points = df[['x','y']].values
    with open('data/route.txt') as f:
        lines = f.readlines()
        paths = [line.strip().split() for line in lines]
        for path in paths:
            path = [int(i) for i in path]
            x = [points[0][0]]
            y = [points[0][1]]
            for p in path:
                x.append(points[p][0])
                y.append(points[p][1])
            x.append(points[0][0])
            y.append(points[0][1])
            sns.lineplot(x=x, y=y, sort=False)
        plt.show()

def main():
    hof, log, paths = ga()
    print(hof.keys)
    print(paths)
    df, vehicle = read_solomon('data/solomon-100/In/c101.txt')
    points = df[['x','y']].values
    real_path = []

    for path in paths:
        x = [points[0][0]]
        y = [points[0][1]]
        for p in path:
            x.append(points[p][0])
            y.append(points[p][1])
        x.append(points[0][0])
        y.append(points[0][1])
        sns.lineplot(x=x,y=y,sort=False)
    plt.show()


if __name__ == '__main__':
    plot_route()

