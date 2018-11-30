import  numpy as np
import  odbc
'''
AGNES层次聚类，采用自底向上聚合策略的算法。先将数据集的每个样本看做一个初始的聚类簇，然后算法运行的每一步中找出距离最近的两个
类簇进行合并，该过程不断重复，直至达到预设的聚类簇的个数。
'''
def agens(dataset,k):
#初始化聚类簇:让每一个点都代表，一个类簇
    sets=[]
    for i in range(0,len(dataset)):
        sets.append({i})
#初始化类簇间距离的矩阵
    delta = np.array(dataset[0] - dataset)
    for e in dataset[1:, :]:
        delta = np.vstack((delta, (e - dataset)))
    distance = np.sqrt(np.sum(np.square(delta), axis=1))
    distance = np.reshape(distance, (len(dataset), len(dataset)))
    distance[np.diag_indices_from(distance)]=float('inf')
    print(distance)
####################################################
    while len(sets)>k:
        locations=np.argwhere(distance==np.min(distance))
        #将集合合并，删除被合并的集合
        locations=locations[locations[:,0]<locations[:,1]]
        cluster_i=locations[0,0]
        cluster_j=locations[0,1]
        for e in sets[cluster_j]:
            sets[cluster_i].add(e)
        del sets[cluster_j]
        #修改distance
        print(sets)

        print("This is last line !")


        print("This is last line !")
        break






dataset=np.loadtxt('data.txt')
agens(dataset,4)