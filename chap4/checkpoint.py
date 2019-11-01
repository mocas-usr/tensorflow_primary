# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:37:08 2019

@author: HAX
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


##原始数据来源
train_x=np.linspace(-1,1,100)
train_y=2*train_x+np.random.randn(*train_x.shape)*0.3


##绘制数据
plt.plot(train_x,train_y,'ro',label='original data')
plt.legend()
plt.show()

##初始化网络结构
X=tf.placeholder('float')
Y=tf.placeholder('float')

##初始化权重变量
w=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.zeros([1]),name='bias')
#超参
learning_rate=0.01
training_epochs=20
plotdata={'batchsize':[],'loss':[]}

#前向传播
z=tf.multiply(X,w)+b
tf.summary.histogram('z',z)##将预测值以直方图显示
##反向传播
cost=tf.reduce_mean(tf.square(Y-z))
tf.summary.scalar('loss_function',cost)
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)



saver=tf.train.Saver(max_to_keep=1)
savedir='log4/'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
            for (x,y) in zip(train_x,train_y):
                sess.run(optimizer,feed_dict={X:x,Y:y})
            if epoch%2==0:
                loss=sess.run(cost,feed_dict={X:train_x,Y:train_y})
                print('epoch:',epoch+1,'loss:',loss,'w=',sess.run(w),'b=',sess.run(b))
                if not(loss=='NA'):
                    plotdata['batchsize'].append(epoch)
                    plotdata['loss'].append(loss)
                saver.save(sess,savedir+'linemodel2.ckpt', global_step=epoch)
            
    print('-------------finish----------')
    loss=sess.run(cost,feed_dict={X:train_x,Y:train_y})
    print('epoch:',epoch+1,'loss:',loss,'w=',sess.run(w),'b=',sess.run(b))
    
    plt.plot(train_x,train_y,'ro',label='orginal data')
    plt.plot(train_x,sess.run(w)*train_x+sess.run(b),label='fitted line')
    plt.legend()
    plt.show()
    
load_epoch=14
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    saver.restore(sess2,savedir+'linemodel2.ckpt-'+str(load_epoch))
    print('x=0.2,z=',sess2.run(z,feed_dict={X:0.2}))
    
    
