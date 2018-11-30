import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import queue
import copy
def createDataset():
    X1, y1 = datasets.make_circles(n_samples=500, factor=.6,
                                   noise=.05)
    X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2, 1.2]], cluster_std=[[.1]],
                                 random_state=9)
    X = np.concatenate((X1, X2))
    # plt.scatter(X[:, 0], X[:, 1], marker='o')
    # plt.show()
    return X
def dbscan(dataset,epsilon,minPts):
    #初始化类簇数为0，未访问样本为数据集D
    ki=0
    unVisit=set(np.arange(0,len(dataset),1))
#1、创建核心对象的集合
    core=set()
    ##计算任意两点间的距离并把结果存到distance矩阵中
    delta=np.array(dataset[0]-dataset)
    for e in dataset[1:,:]:
        delta=np.vstack((delta ,(e-dataset)))
    distance=np.sqrt(np.sum(np.square(delta),axis=1))
    distance=np.reshape(distance,(len(dataset),len(dataset)))

    ##寻找核心对象
    i=0#i记录元素的编号
    for dis in distance:
        if np.count_nonzero(dis<=epsilon) >= minPts:
            core.add(i)
        i+=1
    print("############核心对象已经找到！！！####")
    print(core)
    clusters = []
    while len(core)!=0:
        unVisitOld=copy.copy(unVisit)
        Q=queue.Queue()
        o_index=core.pop()
        unVisit.discard(o_index)
        Q.put(o_index)

        vcore = set()
        #得到一个核心点所有密度可达的点
        while not Q.empty():
            firstEle=Q.get()
            if np.count_nonzero(distance[firstEle] <= epsilon) >= minPts:
                vcore.add(firstEle)
                #得到该核心点，所有密度直达点的编号
                te=distance[firstEle]
                tem=(distance[firstEle]) <= epsilon
                indexs=set(np.argwhere(distance[firstEle]<=epsilon)[:,0])
                delta=indexs & unVisit
                for d in  delta:
                    Q.put(d)
                unVisit=unVisit-delta
        clusters.append(unVisitOld-unVisit)
        core=core-vcore
        vcore = {}

    print("聚类结果如下：")
    print(clusters)
    return clusters,unVisit,len(clusters)


dataset=createDataset()
# dataset=np.loadtxt('data.txt')
print("数据集创建完毕！")
plt.scatter(dataset[:, 0], dataset[:, 1], marker='o')
plt.show()
print("数据集画图完毕！")
# dataset=np.array([[1,2],[1,2.1],[1.1,2],[1,1.9],[0.9,2]])
returnValue,unvist,k=dbscan(dataset,0.15,5)
result=[]
drawdata=[]
result2 = []
drawdata2=[]
id=0
for r  in  returnValue:
    result = []
    for c in r:
        result.append(c)
    drawdata=dataset[result]
    plt.scatter(drawdata[:, 0], drawdata[:, 1], marker='o')
for r  in  unvist:
    result2.append(r)
    drawdata2=dataset[result2]
    plt.scatter(drawdata2[:, 0], drawdata2[:, 1], marker='o')
plt.show()


# print(dataset)

