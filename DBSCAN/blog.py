import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import random
import queue
import copy
#计算两个向量之间的欧式距离
def calDist(X1 , X2 ):
    sum = 0
    for x1 , x2 in zip(X1 , X2):
        sum += (x1 - x2) ** 2
    return sum ** 0.5

#获取一个点的ε-邻域（记录的是索引）
def getNeibor(data , dataSet , e):
    res = []
    for i in range(np.shape(dataSet)[0]):
        if calDist(data , dataSet[i])<e:
            res.append(i)
    return res
def draw(C , dataSet):
    color = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in C.keys():
        X = []
        Y = []
        datas = C[i]
        for j in range(len(datas)):
            X.append(dataSet[datas[j]][0])
            Y.append(dataSet[datas[j]][1])
        plt.scatter(X, Y, marker='o', color=color[i % len(color)], label=i)
    plt.legend(loc='upper right')
    plt.show()

def DBSCAN(dataSet , e , minPts):
    coreObjs = {}#初始化核心对象集合
    C = {}
    n = np.shape(dataSet)[0]
    #找出所有核心对象，key是核心对象的index，value是ε-邻域中对象的index
    for i in range(n):
        neibor = getNeibor(dataSet[i] , dataSet , e)
        if len(neibor)>=minPts:
            coreObjs[i] = neibor
    oldCoreObjs = coreObjs.copy()
    k = 0#初始化聚类簇数
    notAccess = range(n)#初始化未访问样本集合（索引）
    while len(coreObjs)>0:
        OldNotAccess = []
        OldNotAccess.extend(notAccess)
        cores = coreObjs.keys()
        #随机选取一个核心对象
        randNum = random.randint(0,len(cores))
        core = cores[randNum]
        queue = []
        queue.append(core)
        notAccess.remove(core)
        while len(queue)>0:
            q = queue[0]
            del queue[0]
            if q in oldCoreObjs.keys() :
                delte = [val for val in oldCoreObjs[q] if val in notAccess]#Δ = N(q)∩Γ
                queue.extend(delte)#将Δ中的样本加入队列Q
                notAccess = [val for val in notAccess if val not in delte]#Γ = Γ\Δ
        k += 1
        C[k] = [val for val in OldNotAccess if val not in notAccess]
        for x in C[k]:
            if x in coreObjs.keys():
                del coreObjs[x]
    return C

dataSet =createDataset()
C = DBSCAN(dataSet , 0.11 , 5)
draw(C , dataSet)