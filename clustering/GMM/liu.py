import numpy as np
import math
def dim2_gauss(mu,sigma,n,x):
    p=1.0/math.pow(2*np.pi,0.5*n)*math.pow(np.linalg.det(sigma),0.5)
    r=np.dot(np.transpose(x-mu),np.linalg.inv(sigma))
    s=np.dot(r,(x-mu))
    q = math.pow(np.e, (-1 / 2.0) * s)
    return p*q