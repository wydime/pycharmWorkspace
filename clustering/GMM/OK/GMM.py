'''
GMM算法实现，需要修改
'''
import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt
import math
import  random as rd

def create_data(show):
    np.random.seed(0)
    s1=np.random.normal(0,1,(200,2))
    s2=np.random.normal(0,1,(200,2))+np.array([5,3])
    s3=np.random.normal(0,1,(200,2))+np.array([6,10])
    if show==1:
        s=np.vstack((s1,s2,s3))
        plt.plot(s[:,0],s[:,1],'*')
        plt.show()
    s = np.vstack((s1, s2, s3))
    return s
def create_2Dgauss(mu,sigma,sampleNo):
    r=cholesky(sigma)
    s=np.dot(np.random.randn(sampleNo,2),r)+mu
    return s
def calculate_2Dgauss(mu,sigma,x):
    dim=np.shape(sigma)[0]#计算维度
    sigma_det=np.linalg.det(sigma)#计算|∑|
    sigma_Inv= np.linalg.inv(sigma)#计算∑的逆
    d1=(x-mu).T
    d2=(x-mu)
    return np.exp(-0.5 *np.dot(np.dot(d1,sigma_Inv),d2))/(pow(2 * math.pi, dim / 2) * pow(abs(sigma_det), 1 / 2))
# 第一步生成指定mu和sigma的的数据样本，并画出来，生成一个数据集S
mu1=np.array([1,1])
sigma1=np.array([[2,1],[1,2]])
mu2=np.array([5,5])
sigma2=np.array([[2,-1],[-1,2]])
mu3=np.array([8,10])
sigma3=np.array([[4,-2],[-2,4]])
s1=create_2Dgauss(mu1,sigma1,200)
s2=create_2Dgauss(mu2,sigma2,200)
s3=create_2Dgauss(mu3,sigma3,200)
data=np.vstack((s1,s2,s3))
np.savetxt('testdata.txt',data)
#开始GMM算法
#初始化
k     =   3
n     =   data.shape[0]
dim   =   data.shape[1]
alphas =   np.array([0.2,0.2,0.6])
gauss  =   np.ones((n,k))#生成存储，每个点在各个高斯上的权重的矩阵
time   =0
##初始均值为在数据中任意三个点
mus=[0]*k
for i in  range(k):
    mus[i]=data[rd.randint(0,599)]
mus=np.array(mus)
Initmus=mus

sigma=[0]*k
temSigma=np.array([[0.1,0.0],[0.0,0.1]])
sigma[0]=temSigma
sigma[1]=temSigma
sigma[2]=temSigma
sigmas=np.array(sigma)

while time<10:
    # E步
    for i in range(n):#遍历每一个数据
        res=[   alphas[j]*calculate_2Dgauss(mus[j],sigmas[j],data[i]) for j in range(k)]
        sumers=np.sum(res)
        for w in range(k):
            gauss[i][w]=res[w]/sumers
    #M步
    for i in range(k):
        sumline=np.sum(gauss[:,i])
        alphas[i]=sumline/n         #更新比例系数alpha
        #列表表达式
        mus[i]=np.sum([ gauss[j][i] *data[j]  for j in range(n)],axis=0)/sumline#更新均值mus的值
        xdiffs = data - mus[i]
        sigmas[i]=(1/sumline)*np.sum([gauss[nn][i]*xdiffs[nn].reshape(dim,1)*xdiffs[nn] for nn in range(n)],axis=0)
    print(sigmas)
    time+=1
#得到聚类结果
result=np.argmax(gauss,axis=1)
cluster1=data[result==0]
cluster2=data[result==1]
cluster3=data[result==2]

#画图
plt.subplot(121)
plt.plot(s1[:,0],s1[:,1],'+')
plt.plot(s2[:,0],s2[:,1],'+')
plt.plot(s3[:,0],s3[:,1],'+')
plt.subplot(122)
plt.plot(cluster1[:,0],cluster1[:,1],'+')
plt.plot(cluster2[:,0],cluster2[:,1],'+')
plt.plot(cluster3[:,0],cluster3[:,1],'+')
plt.show()




