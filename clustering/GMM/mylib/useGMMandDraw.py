#得到使用第一组高斯混合系数，生成的随机点
import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt
import math
import  random as rd
import OK.gaussPackage as gauss
#第一组样本数据
plt.subplot(221)
data1=np.loadtxt('data1.txt')
plt.plot(data1[:, 0], data1[:, 1], '+')
plt.title('Before Clustering,mixture[0.1,0.1,0.8]')
#第二组样本
plt.subplot(222)
data2=np.loadtxt('data2.txt')
plt.plot(data2[:, 0], data2[:, 1], '+')
plt.title('Before Clustering,mixture[0.1,0.8,0.1]')
#混合样本
plt.subplot(223)
data3=np.loadtxt('datas.txt')
plt.plot(data3[:, 0], data3[:, 1], '+')
plt.title('Before Clustering, sum mixture')
plt.show()
####################################################
####################################################
####################################################
print("比例系数为:[0.1,0.1,0.8]")
c1,c2,c3=gauss.gauss(data1)
plt.subplot(221)
plt.plot(c1[:,0],c1[:,1],'+')
plt.plot(c2[:,0],c2[:,1],'+')
plt.plot(c3[:,0],c3[:,1],'+')
plt.title('After Clustering,mixture[0.1,0.1,0.8]')
print('########################################################')
####################################################
print("比例系数为:[0.1,0.8,0.1]")
c1,c2,c3=gauss.gauss(data2)
plt.subplot(222)
plt.plot(c1[:,0],c1[:,1],'+')
plt.plot(c2[:,0],c2[:,1],'+')
plt.plot(c3[:,0],c3[:,1],'+')
plt.title('After Clustering,mixture[0.1,0.8,0.1]')
print('########################################################')
######################################################
print("两组数据组合成一组：")
c1,c2,c3=gauss.gauss(data3)
plt.subplot(223)
plt.plot(c1[:,0],c1[:,1],'+')
plt.plot(c2[:,0],c2[:,1],'+')
plt.plot(c3[:,0],c3[:,1],'+')
plt.title('After Clustering,mixture')
print('########################################################')
###########################
print('test')
data4=np.loadtxt('testdata.txt')
c1,c2,c3=gauss.gauss(data4)
plt.subplot(224)
plt.plot(c1[:,0],c1[:,1],'+')
plt.plot(c2[:,0],c2[:,1],'+')
plt.plot(c3[:,0],c3[:,1],'+')
plt.title('test')
plt.show()
print('########################################################')
# data=np.loadtxt('datas.txt')
# guss(data)