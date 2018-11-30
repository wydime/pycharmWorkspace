import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl

num = 200
l = np.linspace(-5,5,num)
X, Y = np.meshgrid(l, l) #meshgrid的作用适用于生成网格型数据，可以接受两个一维数组生成两个二维矩阵
#np.expand_dims增加一个维度(下面是增加第三维)
pos = np.concatenate((np.expand_dims(X,axis=2),np.expand_dims(Y,axis=2)),axis=2)

def plot_multi_normal(u,sigma):
    fig = plt.figure(figsize=(12,7))
    ax = Axes3D(fig)

    a = (pos-u).dot(np.linalg.inv(sigma))   #np.linalg.inv()矩阵求逆
    b = np.expand_dims(pos-u,axis=3)
    Z = np.zeros((num,num), dtype=np.float32)
    for i in range(num):
        Z[i] = [np.dot(a[i,j],b[i,j]) for j in range(num)]
    Z = np.exp(Z*(-0.5))/(2*np.pi*(np.linalg.det(sigma))**(0.5))   #np.linalg.det()矩阵求行列式
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5, alpha=0.4, cmap=mpl.cm.bwr)
    cset = ax.contour(X,Y,Z, zdir='z',offset=0,cmap=cm.coolwarm,alpha=0.8)  #contour画等高线
    cset = ax.contour(X, Y, Z, zdir='x', offset=-5,cmap=mpl.cm.winter,alpha=0.8)
    cset = ax.contour(X, Y, Z, zdir='y', offset= 5,cmap= mpl.cm.winter,alpha=0.8)
    ax.set_zlim([0,0.3])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


u = np.array([0, 0])
sigma = np.array([[1, 0],[0, 1]])
plot_multi_normal(u,sigma)