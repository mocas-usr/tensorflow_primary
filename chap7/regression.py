# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:47:46 2019

@author: HAX
"""

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
#模拟数据点函数 
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

input_dim=2
np.random.seed(10)
num_classes=2
mean=np.random.randn(num_classes)
cov=np.eye(num_classes)
X,Y=generate(1000,mean,cov,[3.0],True)
colors = ['r' if l == 0 else 'b' for l in Y[:]]
plt.scatter(X[:,0], X[:,1], c=colors)
plt.xlabel('scaled age (in yrs)')
plt.ylabel('tumor size (in cm)')
plt.show()
lab_dim=1


input_features=tf.placeholder(tf.float32,[None,input_dim])
input_labels=tf.placeholder(tf.float32,[None,lab_dim])

#定义学习参数
w=tf.Variable(tf.random_normal([input_dim,lab_dim]),name='weight')
b=tf.Variable(tf.zeros([lab_dim]))

output=tf.nn.sigmoid(tf.matmul(input_features,w)+b)
cross_entroy=-(input_labels*tf.log(output)+(1-input_labels)*tf.log(1-output))
ser=tf.square(input_labels-output)
loss=tf.reduce_mean(cross_entroy)
err=tf.reduce_mean(ser)
optimizer=tf.train.AdamOptimizer(0.04)
train=optimizer.minimize(loss)

##设置参数训练
training_size=len(Y)

maxepoch=50
minibatchsize=25
#启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ##向模型输入数据
    for epoch in range(maxepoch):
        sumerr=0
        for i in range(np.int32(len))