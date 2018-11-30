import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy.linalg import cholesky

def my_dis(a,s):
    dis=np.ones(len(s))
    j=0
    for i in s:
        dis[j]=np.sqrt(np.square((i[0]-a[0]))+np.square((i[1]-a[1])))
        j=j+1
    return dis
SampleAmount=200
s1=np.random.normal(0,1,(SampleAmount,2))
s2=np.random.normal(0,1,(SampleAmount,2))+np.array([5,5])
s3=np.random.normal(0,1,(SampleAmount,2))+np.array([8,8])
s=np.vstack((s1,s2,s3))

plt.plot(s1[:,0],s1[:,1],'*')
plt.plot(s2[:,0],s2[:,1],'o')
plt.plot(s3[:,0],s3[:,1],'^')
plt.show()


k=200
# K3=np.array([[0,0.5],[4,4.3],[7,7.2]])
a=[0,0.5]
for i in range(10):
    dis = my_dis(a,s)
    a=np.mean(s[dis.argsort()[0:k],:],axis=1)


