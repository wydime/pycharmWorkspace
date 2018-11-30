# -*- coding: utf-8 -*-

'''
 Description ：GMM in Python
 Author ： LiuLongpo
 Time : 2015年7月26日16:54:48
 Source ：From pluskid
'''

import sys

import matplotlib.pyplot as plt
import numpy as np
import random
'矩阵的逆矩阵需要的库'
from numpy.linalg import *
def create_2Dgauss(mu,sigma,sampleNo):
    r=cholesky(sigma)
    s=np.dot(np.random.randn(sampleNo,2),r)+mu
    return s
def calculate_2Dgauss(mu,sigma,x):
    dim=np.shape(sigma)[0]#计算维度
    print(sigma)
    sigma_det=np.linalg.det(sigma)#计算|∑|
    sigma_Inv= np.linalg.inv(sigma)#计算∑的逆
    d1=(x-mu).T
    d2=(x-mu)
    return np.exp(-0.5 *np.dot(np.dot(d1,sigma_Inv),d2))/(pow(2 * math.pi, dim / 2) * pow(abs(sigma_det), 1 / 2))

def gmm(X,K):
    threshold  = 1e-15
    N,D = np.shape(X)
    # randV = pu.randIntList(1,N,K)
    randV =np.random.randint(1,N,K)
    centroids = X[randV]
    pMiu,pPi,pSigma = inti_params(centroids,K,X,N,D);
    Lprev = -np.inf
    while True:
        'Estiamtion Step'
        Px = calc_prop(X,N,K,pMiu,pSigma,threshold,D)
        pGamma = Px * np.tile(pPi,(N,1))
        pGamma = pGamma / np.tile((np.sum(pGamma,axis=1)),(K,1)).T
        'Maximization Step'
        Nk = np.sum(pGamma,axis=0)
        pMiu = np.dot(np.dot(np.diag(1 / Nk),pGamma.T),X)
        pPi = Nk / N
        for kk in range(K):
            Xshift = X - np.tile(pMiu[kk],(N,1))
            pSigma[:,:,kk] = (np.dot(np.dot(Xshift.T,np.diag(pGamma[:,kk])),Xshift)) / Nk[kk]

        'check for convergence'
        L = np.sum(np.log(np.dot(Px,pPi.T)))
        if L-Lprev<threshold:
            break
        Lprev = L

    return Px

def inti_params(centroids,K,X,N,D):
    pMiu = centroids
    pPi = np.zeros((1,K))
    pSigma = np.zeros((D,D,K))
    distmat = np.tile(np.sum(X * X,axis=1),(K,1)).T \
    + np.tile(np.sum(pMiu * pMiu,axis = 1).T,(N,1)) \
    - 2 * np.dot(X,pMiu.T)
    labels = np.argmin(distmat,axis=1)

    for k in range(K):
        Xk = X[labels==k]
        pPi[0][k] = float(np.shape(Xk)[0]) / N # 样本数除以 N 得到概率
        pSigma[:,:,k] = np.cov(Xk.T)
    return pMiu,pPi,pSigma

    '计算概率'
def calc_prop(X,N,K,pMiu,pSigma,threshold,D):
    Px = np.zeros((N,K))
    for k in range(K):
        Xshift = X - np.tile(pMiu[k],(N,1))
        inv_pSigma = inv(pSigma[:,:,k]) \
        + np.diag(np.tile(threshold,(1,np.ndim(pSigma[:,:,k]))))
        tmp = np.sum(np.dot(Xshift,inv_pSigma) * Xshift,axis=1)
        coef = (2*np.pi)**(-D/2) * np.sqrt(np.linalg.det(inv_pSigma))
        Px[:,k] = coef * np.exp(-0.5 * tmp)
    return Px

def test():
    # 第一步生成指定mu和sigma的的数据样本，并画出来，生成一个数据集S
    mu1 = np.array([1, 1])
    sigma1 = np.array([[2, 1], [1, 2]])
    mu2 = np.array([5, 5])
    sigma2 = np.array([[2, -1], [-1, 2]])
    mu3 = np.array([8, 10])
    sigma3 = np.array([[4, -2], [-2, 4]])
    s1 = create_2Dgauss(mu1, sigma1, 200)
    s2 = create_2Dgauss(mu2, sigma2, 200)
    s3 = create_2Dgauss(mu3, sigma3, 200)
    # 画图
    # plt.subplot(121)
    # plt.plot(s1[:, 0], s1[:, 1], '+')
    # plt.plot(s2[:, 0], s2[:, 1], '+')
    # plt.plot(s3[:, 0], s3[:, 1], '+')
    # plt.show()
    X=np.vstack((s1,s2,s3))
    num = np.size(X)
    X = np.reshape(X,(num/2,2))
    ppx = gmm(X,4)
    index = np.argmax(ppx,axis=1)
    plt.figure()
    plt.scatter(X[index==0][:,0],X[index==0][:,1],s=60,c=u'r',marker=u'o')
    plt.scatter(X[index==1][:,0],X[index==1][:,1],s=60,c=u'b',marker=u'o')
    plt.scatter(X[index==2][:,0],X[index==2][:,1],s=60,c=u'y',marker=u'o')
    plt.scatter(X[index==3][:,0],X[index==3][:,1],s=60,c=u'g',marker=u'o')


if __name__ == '__main__':

    test()
