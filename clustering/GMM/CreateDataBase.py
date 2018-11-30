# coding=utf-8

import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
sampleNo = 1000;
# 一维正态分布
mu = 3
sigma = 0.1
np.random.seed(0)
s = np.random.normal(mu, sigma, sampleNo)
plt.subplot(141)#一行两列的第一个位置
plt.hist(s,30, density=True)
# 二维正态分布
mu = np.array([[1, 50]])
Sigma = np.array([[4, 3], [3, 4]])
R = cholesky(Sigma)
s = np.dot(np.random.randn(sampleNo, 2), R) + mu
plt.subplot(142)
# 注意绘制的是散点图，而不是直方图
plt.plot(s[:,0],s[:,1],'+')
plt.show()
###################################
# a=np.random.normal(0,1,(50,2))
# plt.plot(s[:,0],a[:,1],'+')
# plt.show()
np.dot(R,R.T)