# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 19:59:40 2019

@author: HAX
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


##模拟数据
train_x=np.linspace(-1,1,100)
train_y=2*train_x+np.random.randn(*train_x.shape)*0.2
plt.plot(train_x,train_y,'ro',label='orginal data ')
plt.legend()
plt.show()

##重置图
tf.reset_default_graph()

##初始网络结构
X=tf.placeholder('float')
Y=tf.placeholder('float')
#初始化权重
w=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.zeros([1],name='bias'))
#前向传播
z=tf.multiply(X,w)+b

#初始化超参
learning_rate=0.01
training_epochs=20

#反向传播
cost=tf.reduce_mean(tf.square(Y-z))
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


##saver生成
saver=tf.train.Saver()
savedir='log/'##生成路径
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        for (x,y) in zip(train_x,train_y):
            sess.run(optimizer,feed_dict={X:train_x,Y:train_y})
        
    print('finish------------------')
    saver.save(sess,savedir+'linemodel.ckpt')
    print('cost=',sess.run(cost,feed_dict={X:train_x,Y:train_y}),'w=',sess.run(w),'b=',sess.run(b))
    


with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    saver.restore(sess2,savedir+'linemodel.ckpt')
    print('x=0.2,z=',sess2.run(z,feed_dict={X:0.2}))




from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file as prtensor

savedir='log/'
prtensor(savedir+'linemodel.ckpt',None,True)



