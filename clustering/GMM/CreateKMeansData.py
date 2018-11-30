import numpy as np
from numpy.linalg import cholesky
import  matplotlib.pyplot as plt
import random
'''
自定义的函数
'''
#计算距离
def  get_dis(vec1,vec2):
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return dist
#随机生成聚类中心  K代表生随机生成的个数
def rand_center(dataset,k):
    indexList=[]
    returnList=[]

    while len(indexList)!=k:
        randInt=random.randint(0,len(dataset)-1)
        if randInt in indexList:
            continue
        else:
            indexList.append(randInt)
    for i in indexList:
        returnList.append(dataset[i])
    return returnList

'''
生成随机的二维 正态分布
'''
def create_data():
    np.random.seed(0)
    s1=np.random.normal(0,1,(200,2))
    s2=np.random.normal(0,1,(200,2))+np.array([5,3])
    s3=np.random.normal(0,1,(200,2))+np.array([6,10])
    # plt.plot(s1[:,0],s1[:,1],'*')
    # plt.plot(s2[:,0],s2[:,1],'o')
    # plt.plot(s3[:,0],s3[:,1],'^')
    # plt.show()
    s=np.vstack((s1,s2,s3))
    plt.plot(s[:,0],s[:,1],'*')
    plt.show()
    return s
'''
K-Means方法实现聚类
'''
def K_Means():
    data=create_data()
    centers,b=rand_center(data, 3)
    dist=[]
    for i in range(0,len(data)-1):
        for center in centers:
            dist.append(get_dis(data[i],center))
        min()

    print(1)

K_Means()

a=np.array([[1],[2],[3]])














