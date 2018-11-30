import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt
import math
import  random
'''
创建二维随机正态分布样本点
mu          均值（决定中心位置，改变mu意味着中心移动）
sigma       协方差矩阵（决定概率密度的敏感度）
sampleNO    样本数目
'''

def create_2Dgauss(mu, sigma, sampleNo):
    r = cholesky(sigma)
    s = np.dot(np.random.randn(sampleNo, 2), r) + mu
    return s

"按照权重生成随机数"
'''
list   目标列表  类型（list）
weight 权重列表  类型（list）
N      数目      类型（int） 
'''


def weight_random(list, weight, N):
    R = []
    weight = np.array(weight)
    weight = weight / sum(weight)
    weight = np.cumsum(weight)
    for n in range(N):
        rndint = random.random()
        for i in range(len(weight)):
            if rndint < weight[i]:
                R.append(list[i])
                break

list=['A','B','C']
R = weight_random(list, [0.3, 0.1, 0.6], 100000)

all = []
# for i in range(100000):
#     all.append(np.random.choice(list, p=[0.3, 0.2, 0.5]))
all=(np.random.choice(list,1000,p=[0.3,0.2,0.5])).tolist()
print("a的概率", all.count('A') / len(all))
print("b的概率", all.count('B') / len(all))
print("c的概率", all.count('C') / len(all))