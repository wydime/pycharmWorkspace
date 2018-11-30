import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt
import math
import  random

def create_2Dgauss(mu,sigma,sampleNo):
    r=cholesky(sigma)
    s=np.dot(np.random.randn(sampleNo,2),r)+mu
    return s
###三个高斯的均值和协方差矩阵参数
mu1=np.array([1,1])
sigma1=np.array([[2,1],[1,2]])
mu2=np.array([5,5])
sigma2=np.array([[2,-1],[-1,2]])
mu3=np.array([8,10])
sigma3=np.array([[4,-2],[-2,4]])
####根据不同的混合比例比例，随机生成选择哪个高斯
#gaussNo[]  存储第几个高斯，生成随机点
list=[0,1,2]
gaussNo1=[]
gaussNo2=[]
gaussNo1=(np.random.choice(list,1000,p=[0.1,0.1,0.8])).tolist()
gaussNo2=(np.random.choice(list,1000,p=[0.1,0.8,0.1])).tolist()
#得到使用第一组高斯混合系数，生成的随机点
s1=create_2Dgauss(mu1,sigma1,gaussNo1.count(0))
s2=create_2Dgauss(mu2,sigma2,gaussNo1.count(1))
s3=create_2Dgauss(mu3,sigma3,gaussNo1.count(2))
plt.subplot(221)
plt.plot(s1[:,0],s1[:,1],'+')
plt.plot(s2[:,0],s2[:,1],'+')
plt.plot(s3[:,0],s3[:,1],'+')
s=np.vstack((s1,s2,s3))
np.savetxt('data1.txt',s)
#得到使用第二组高斯混合系数，生成的随机点
ss1=create_2Dgauss(mu1,sigma1,gaussNo2.count(0))
ss2=create_2Dgauss(mu2,sigma2,gaussNo2.count(1))
ss3=create_2Dgauss(mu3,sigma3,gaussNo2.count(2))
plt.subplot(222)
plt.plot(ss1[:,0],ss1[:,1],'+')
plt.plot(ss2[:,0],ss2[:,1],'+')
plt.plot(ss3[:,0],ss3[:,1],'+')
ss=np.vstack((ss1,ss2,ss3))
np.savetxt('data2.txt',ss)
#将两组不同混合比例的数据，放到一起画出来
mixture=np.vstack((s1,s2,s3,ss1,ss2,ss3))
plt.subplot(223)
plt.plot(mixture[:,0],mixture[:,1],'+')
plt.show()
###将生成的样本保存到txt文件
np.savetxt('datas.txt',mixture)