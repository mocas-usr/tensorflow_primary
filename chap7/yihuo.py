# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:30:16 2019

@author: HAX
"""

import tensorflow as tf
import numy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


#模拟数据点   
def generate(sample_size, mean, cov, diff,regression):   
    num_classes = 2 #len(diff)
    samples_per_class = int(sample_size/2)

    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)
    
    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean+d, cov, samples_per_class)
        Y1 = (ci+1)*np.ones(samples_per_class)
    
        X0 = np.concatenate((X0,X1))
        Y0 = np.concatenate((Y0,Y1))
        
    if regression==False: #one-hot  0 into the vector "1 0
        print("ssss")
        class_ind = [Y0==class_number for class_number in range(num_classes)]
        Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    X, Y = shuffle(X0, Y0)
    
    return X,Y
np.random.seed(10)
input_dim=2
num_classes=4
X,Y=generate(320,num_classes,[[3.0,0],[3.0,3.0],[0.3,0]],True)
Y=Y%2

xr=[]
xb=[]
for (l,k) in zip(Y[:],X[:]):
    if l==0.0:
        xr.append(k[0],k[1])
    else:
        xb.append(k[0],k[1])
    
xr=np.array(xr)
xb=np.array(xb)
plt.scatter(xr[:,0],)

