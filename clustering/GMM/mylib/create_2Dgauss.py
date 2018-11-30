import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt
import math
'''
创建二维随机正态分布样本点
mu          均值（决定中心位置，改变mu意味着中心移动）
sigma       协方差矩阵（决定概率密度的敏感度）
sampleNO    样本数目
'''
def create_2Dgauss(mu,sigma,sampleNo):
    r=cholesky(sigma)
    s=np.dot(np.random.randn(sampleNo,2),r)+mu
    return s
'''
计算高斯函数
mu          均值向量
sigma       协方差矩阵
x           数据
n           n元高斯
'''

def calculate_2Dgauss(mu,sigma,x):
    dim=np.shape(sigma)[0]#计算维度
    sigma_det=np.linalg.det(sigma)#计算|∑|
    sigma_Inv= np.linalg.inv(sigma)#计算∑的逆
    d1=(x-mu).T
    d2=(x-mu)
    return np.exp(-0.5 *np.dot(np.dot(d1,sigma_Inv),d2))/(pow(2 * math.pi, dim / 2) * pow(np.linalg.det(sigma), 1 / 2))
# x1=np.array([2,1])
mu1=np.array([0,1])
sigma1=np.array([[2,1],[1,2]])

plt.plot(s3[:,0],s3[:,1],'+')
plt.show()
# print(calculate_2Dgauss(mu,sigma,x,))
# print(calculate_2Dgauss(mu,sigma,x,))

