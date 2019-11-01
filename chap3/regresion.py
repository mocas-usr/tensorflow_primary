# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 09:23:50 2019

@author: HAX
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def moving_average(a,w=10):
    if len(a)<w:
        return a[:]
    return [val if idx<w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]
#
train_x=np.linspace(-1,1,100)
train_y=2*train_x+np.random.randn(*train_x.shape)*0.3
plt.plot(train_x,train_y,'ro',label='data_plot')
plt.legend()
plt.show()

#创建模型
X=tf.placeholder('float')
Y=tf.placeholder('float')
w=tf.Variable(tf.random_normal([1]),name='weight')#一维张量
b=tf.Variable(tf.zeros([1]),name='bias')
#前项结构
z=tf.multiply(X,w)+b
#反向优化
cost=tf.reduce_mean(tf.square(Y-z))
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
##初始化变量
init=tf.global_variables_initializer()
#定义参数
training_epochs=20
display_step=2

##启动session
with tf.Session() as sess:
    sess.run(init)
    plotdata={'batchsize':[],'loss':[]}
    for epoch in range(training_epochs):
        for (x,y) in zip(train_x,train_y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
        
        ##显示训练的详细信息
        if epoch%display_step==0:
            loss=sess.run(cost,feed_dict={X:train_x,Y:train_y})
            print('epoch',epoch+1,'cost=',loss,'w=',sess.run(w),'b=',sess.run(b))
            if not(loss=='NA'):
                plotdata['batchsize'].append(epoch)    
                plotdata['loss'].append(loss)
                
    print('finish')
    print('cost=',sess.run(cost,feed_dict={X:train_x,Y:train_y}),'w=',sess.run(w),'b=',sess.run(b))
    
    ##图像显示
    plt.plot(train_x,train_y,'ro',label='original_data')
    plt.plot(train_x,sess.run(w)*train_x+sess.run(b),label='fittedline')
    plt.legend()
    plt.show()
    plotdata['avgloss']=moving_average(plotdata['loss'])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata['batchsize'],plotdata['avgloss'],'b--')
    plt.xlabel('minibatch number')
    plt.ylabel('loss')
    plt.title('minibatch run vs traiining loss')
    plt.show()
    
    
            

