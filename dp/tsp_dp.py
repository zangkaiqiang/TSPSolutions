# 使用状态压缩的动态规划
import numpy as np
import math
import time
np.random.seed(0)

# 计算二进制数中含有1的个数
def count_one(n):
    if n < 1:
        return 0
    one = 0
    while n > 0:
        if n%2==1:
            one = one + 1
        n = n >> 1
    return one

# 点的总个数
n = 10
# 需要点的个数
m = 10

# 随机生成n个点 p(x,y)
x = np.random.random(n) * 100
y = np.random.random(n) * 100

# 距离矩阵，计算每两个点的距离
dis = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        pi = (x[i],y[i])
        pj = (x[j],y[j])
        dij = math.sqrt((x[j]-x[i])**2 + (y[j]-y[i])**2)
        dis[i,j] = dij

# 状态个数
state = 1 << n
# 每种状态的最短路径矩阵，初始化为一个较大的数
dp = np.zeros((state, n)) + math.inf
# 保存每种状态的路径信息
path = []
for i in range(state):
    path_state = []
    for j in range(n):
        path_state.append([])
    path.append(path_state)

# 设定起始点为第0个点，起始状态到0的距离为0
dp[1,0] = 0

# 保存含有m个点的状态数组
key = []

start_time = time.time()
# dp过程
# 从初始状态1开始，遍历每个状态
for i in range(1, state):
    # 跳过出发点不是0的状态
    if (i & 1 == 0):
        continue
    # 计算状态i中含有的点的个数
    one = count_one(i)
    # 保存个数为m的点
    if one == m:
        key.append(i)

    # 如果状态i中含有的点的个数大于m，则不计算该状态
    if one > m:
        continue

    # 选取0以外的点
    for j in range(1, n):
        # 如果点j在状态i中，则跳过j选择下一个点
        if (i & (1<<j)):
            continue
        # 加入j点，更新新状态下到j的最短距离
        for k in range(n):
            if (i & (1<<k)): # 判断k点是否在状态i中
                # 更新最短距离
                # 更新path
                if dp[(1 << j) | i, j] > (dp[i, k] + dis[k, j]):
                    dp[(1 << j) | i, j] = dp[i, k] + dis[k, j]

                    path[(1 << j) | i][j] = path[(1 << k) | i][k].copy()
                    path[(1 << j) | i][j].append(j)

print('use seconds:',time.time()-start_time)

# 找到最短距离
min_din = math.inf
min_state = -1
min_index = -1
for i in key:
    if min(dp[i]) < min_din:
        min_din = min(dp[i])
        min_state = i
        min_index = dp[i].argmin()

print(min_din)
# 输出path不包含起始点0
print(path[min_state][min_index])

