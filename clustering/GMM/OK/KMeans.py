import numpy as np
from numpy.linalg import cholesky
import  matplotlib.pyplot as plt
import random
def  get_dis(vec1,vec2):
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return dist
def rand_center(dataset,k):
    indexList=[]
    returnList=[]

    while len(indexList)!=k:
        randInt=random.randint(0,len(dataset)-1)
        if randInt in indexList:
            continue
        else:
            indexList.append(randInt)
    for i in indexList:
        returnList.append(dataset[i])
    return returnList,indexList
np.random.seed(0)
s1=np.random.normal(0,1,(200,2))
s2=np.random.normal(0,1,(200,2))+np.array([3,3])
s3=np.random.normal(0,1,(200,2))+np.array([5,10])

s=np.vstack((s1,s2,s3))
# plt.subplot(121)
# plt.plot(s[:,0],s[:,1],'*')
plt.show()
center,index=rand_center(s,3)
centers=np.array(center)
time=0
distance_mat=np.ones((len(s),3))
while time<6:
#计算距离
    for i in range(0,600):
        for j in range(0,3):
            distance_mat[i][j]=get_dis(s[i],centers[j])
#得到每个点的分类到的类簇
    flag=np.argmin(distance_mat,axis=1)
#更新每个点的中心
    centers[0]=np.mean(s[(flag==0)],axis=0)
    centers[1]=np.mean(s[(flag==1)],axis=0)
    centers[2]=np.mean(s[(flag==2)],axis=0)
    time+=1

plt.subplot(221)
plt.plot(s[:,0],s[:,1],'*')
C1=s[(flag==0)]
C2=s[(flag==1)]
C3=s[(flag==2)]
plt.subplot(222)
plt.plot(C1[:,0],C1[:,1],'*')
plt.plot(C2[:,0],C2[:,1],'o')
plt.plot(C3[:,0],C3[:,1],'^')
plt.show()






