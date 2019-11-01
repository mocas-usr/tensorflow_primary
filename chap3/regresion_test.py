# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 15:14:10 2019

@author: HAX
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


##制作数据
x=np.linspace(-1,1,100)
train_x=x
train_y=train_x*2+np.random.randn(*train_x.shape)*0.2
plt.plot(train_x,train_y,'rs',label='original data')

#构建网络结构
#初始化输入量
X=tf.placeholder('float')
Y=tf.placeholder('float')
##初始化权重变量
w=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.zeros([1]),name='bias')


##前向传播
z=tf.multiply(X,w)+b

##反向优化

cost=tf.reduce_mean(tf.square(Y-z))
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

##训练模型
init=tf.global_variables_initializer()
training_epochs=20#训练次数
display_step=2##打印间隔

##启动Session

with tf.Session() as sess:
    sess.run(init)
    plotdata={'batchsize':[],'loss':[]}#用于存放批的数值，和损失值
    
    ##向模型输入数据
    for epoch in range(training_epochs):
        for (x,y) in zip(train_x,train_y):
            sess.run(optimizer,feed_dict={X:train_x,Y:train_y})
        
        
        if epoch%display_step==0:
            loss=sess.run(cost,feed_dict={X:train_x,Y:train_y})
            print('epoch:',epoch+1,'cost:',loss,'w=',sess.run(w),'b=',sess.run(b))
            if not(loss=='NA'):
                plotdata['batchsize'].append(epoch)
                plotdata['loss'].append(loss)
            
    print('finish')
    print('cost=',sess.run(cost,feed_dict={X:train_x,Y:train_y}),'w=',sess.run(w),'b=',sess.run(b))
    

    

