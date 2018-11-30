import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt
import math
def calculate_2Dgauss(mu,sigma,x,n):
    a=np.exp(-0.5*np.dot(np.dot(  (x - mu).T  ,  np.linalg.inv(sigma)  ),(x-mu)))#分子
    b =1/(pow(2 * math.pi, n / 2) * pow(np.linalg.det(sigma), 1 / 2))  # 分母
    return a*b

def Gaussian(data,mean,cov):
    dim = np.shape(cov)[0]   # 计算维度
    covdet = np.linalg.det(cov) # 计算|cov|
    covinv = np.linalg.inv(cov) # 计算cov的逆
    if covdet==0:              # 以防行列式为0
        covdet = np.linalg.det(cov+np.eye(dim)*0.01)
        covinv = np.linalg.inv(cov+np.eye(dim)*0.01)
    m = data - mean
    z = -0.5 * np.dot(np.dot(m, covinv),m)    # 计算exp()里的值
    return 1.0/(np.power(np.power(2*np.pi,dim)*abs(covdet),0.5))*np.exp(z)  # 返回概率密度值
x=np.array([2,1])
mu=np.array([1,5])
sigma=np.array([[5,-3],[-3,5]])

print(Gaussian(x,mu,sigma))
