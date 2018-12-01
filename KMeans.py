# -*- coding: utf-8 -*-
import numpy as np
import  matplotlib.pyplot as plt
import random
def  get_dis(X1,X2):
    # dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    # return dist
    sum = 0
    for x1, x2 in zip(X1, X2):
        sum += (x1 - x2) ** 2
    return sum ** 0.5
def  rand_center(dataset,k):
    index=[]
    index=random.sample(range(1,len(dataset)),k)#range()左闭右开
    # return dataset[index]
    return dataset[[5, 11, 23]]
'''
dataset数据集
k聚类的个数
n迭代的次数
'''
def k_means(dataset,k,n):
    centers = rand_center(dataset, k)
    #distance矩阵存储，每个点到中心点的距离
    distance_mat = np.ones((len(dataset), k))
    time=0
    flag = []
    while time < n:
        # 计算距离
        for i in range(0, len(dataset)):
            for j in range(0, k):
                distance_mat[i][j] = get_dis(dataset[i], centers[j])
        # 用distance_mat矩阵中的每一行最小值的下标，作为分类的标记。得到每个点的分类到的类簇，
        flag = np.argmin(distance_mat, axis=1)
        # 更新每个点的中心
        centers[0] = np.mean(dataset[flag == 0], axis=0)
        centers[1] = np.mean(dataset[flag == 1], axis=0)
        centers[2] = np.mean(dataset[flag == 2], axis=0)
        # i=0
        # for center in centers:
        #     centers[center==centers]=np.mean(dataset[flag==i], axis=0)
        #     i+=1
        time += 1

    return  flag
def draw_result(dataset,flag):
    #画出未分类前的数据
    # plt.subplot(221)
    # plt.plot(dataset[:, 0], dataset[:, 1], '*')
    # plt.title('未分类前的数据集')
    # plt.xlabel('密度')
    # plt.ylabel('含糖率')
    #画出聚类后的数据
    # plt.subplot(222)
    C1 = dataset[(flag == 0)]
    C2 = dataset[(flag == 1)]
    C3 = dataset[(flag == 2)]
    plt.plot(C1[:, 0], C1[:, 1], '*')
    plt.plot(C2[:, 0], C2[:, 1], 'o')
    plt.plot(C3[:, 0], C3[:, 1], '^')
    plt.show()

dataset=np.loadtxt('data.txt')
result=k_means(dataset,3,1)
draw_result(dataset,result)
# np.random.seed(0)
# s1=np.random.normal(0,1,(200,2))
# s2=np.random.normal(0,1,(200,2))+np.array([3,3])
# s3=np.random.normal(0,1,(200,2))+np.array([5,10])
# s=np.vstack((s1,s2,s3))
# # plt.subplot(121)
# # plt.plot(s[:,0],s[:,1],'*')
# plt.show()
# # center,index=rand_center(s,3)
# s=np.loadtxt('data.txt')
# center=rand_center(s,3)
# centers=np.array(center)
# time=0
# distance_mat=np.ones((len(s),3))
# while time<5:
# #计算距离
#     for i in range(0,len(s)):
#         for j in range(0,3):
#             distance_mat[i][j]=get_dis(s[i],centers[j])
# #得到每个点的分类到的类簇
#     flag=np.argmin(distance_mat,axis=1)
# #更新每个点的中心
#     centers[0]=np.mean(s[(flag==0)],axis=0)
#     centers[1]=np.mean(s[(flag==1)],axis=0)
#     centers[2]=np.mean(s[(flag==2)],axis=0)
#     time+=1
#




