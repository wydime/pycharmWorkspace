import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt
import math
import  random as rd
def calculate_2Dgauss(mu,sigma,x):
    dim=np.shape(sigma)[0]#计算维度
    sigma_det=np.linalg.det(sigma)#计算|∑|
    sigma_Inv= np.linalg.inv(sigma)#计算∑的逆
    if sigma_det   ==  0:
        sigma_det=np.linalg.det(sigma_Inv+np.eye(dim)*0.01)
        sigma_Inv=np.linalg.inv(sigma_Inv+np.eye(dim)*0.01)
    d1=(x-mu).T
    d2=(x-mu)
    return np.exp(-0.5 *np.dot(np.dot(d1,sigma_Inv),d2))/(pow(2 * math.pi, dim / 2) * pow(abs(sigma_det), 1 / 2))
def gauss(data):
    # 开始GMM算法
    # 初始化
    k = 3
    n = data.shape[0]
    dim = data.shape[1]
    alphas = np.array([0.3, 0.3, 0.3])
    gauss = np.ones((n, k))  # 生成存储，每个点在各个高斯上的权重的矩阵
    time = 0
    ##初始均值为在数据中任意三个点
    mus = [0] * k
    for i in range(k):
        mus[i] = data[rd.randint(0, 599)]
    mus = np.array(mus)
    Initmus = mus

    sigma = [0] * k
    for i in range(k):
        # 初始的协方差矩阵源自于原始数据的协方差矩阵，且每个簇的初始协方差矩阵相同
        sigma[i] = np.cov(data.T)
    sigmas = np.array(sigma)

    while time < 50:
        # E步
        for i in range(n):  # 遍历每一个数据
            res = [alphas[j] * calculate_2Dgauss(mus[j], sigmas[j], data[i]) for j in range(k)]
            sumers = np.sum(res)
            for w in range(k):
                gauss[i][w] = res[w] / sumers
        # M步
        for i in range(k):
            sumline = np.sum(gauss[:, i])
            alphas[i] = sumline / n  # 更新比例系数alpha
            # 列表表达式
            mus[i] = np.sum([gauss[j][i] * data[j] for j in range(n)], axis=0) / sumline  # 更新均值mus的值
            xdiffs = data - mus[i]
            sigmas[i] = (1 / sumline) * np.sum(
                [gauss[nn][i] * xdiffs[nn].reshape(dim, 1) * xdiffs[nn] for nn in range(n)], axis=0)
        time += 1
    # 得到聚类结果
    result = np.argmax(gauss, axis=1)
    cluster1 = data[result == 0]
    cluster2 = data[result == 1]
    cluster3 = data[result == 2]
    print('混合比例系数为：\n\n\t',alphas)
    print('均值为：      ：\n\n\t',mus)
    print('协方差矩阵为  ：\n\n\t',sigmas)
    return cluster1,cluster2,cluster3
    # # 画图
    # plt.subplot(122)
    # plt.plot(cluster1[:, 0], cluster1[:, 1], '+')
    # plt.plot(cluster2[:, 0], cluster2[:, 1], '+')
    # plt.plot(cluster3[:, 0], cluster3[:, 1], '+')
    # plt.show()
# data=np.loadtxt('datas.txt')
# gauss(data)