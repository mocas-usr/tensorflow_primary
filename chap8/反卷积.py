# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:00:22 2019

@author: HAX
"""

import tensorflow as tf
import numpy as np

##模拟数据
img=tf.Variable(tf.constant(1.0,shape=[1,4,4,1]))
filter=tf.Variable(tf.constant([1.0,0,-1,-2],shape=[2,2,1,1]))
##f分别进行VALID 和SAME 操作
conv1=tf.nn.conv2d(img,filter,strides=[1,2,2,1],padding='SAME')
conv2=tf.nn.conv2d(img,filter,strides=[1,2,2,1],padding='VALID')
print(conv1.shape)
print(conv2.shape)
##再进行反卷积
contv1=tf.nn.conv2d_transpose(conv1,filter,[1,4,4,1],strides=[1,2,2,1],padding='VALID')
contv2=tf.nn.conv2d_transpose(conv2,filter,[1,4,4,1],strides=[1,2,2,1],padding='SAME')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('conv:\n',sess.run([conv1,filter]))
    print('conv2',sess.run(conv2))
    print('contv1',sess.run(contv1))
    print('contv2',sess.run(contv2))
    
    

def max_pool_with(net,stride):
    _,mask=tf.nn.max_pool_with_argmax(net,ksize=[1,stride,stride,1],strides=[1,stride,stride,1],padding='SAME')
    mask=tf.stop_gradient(mask)
    net=tf.nn.max_pool(net,ksize=[1,stride,stride,1],strides=[1,stride,stride,1],padding='SAME')
    return net,mask



import tensorflow as tf
w1=tf.Variable([[1.,2]])
w2=tf.Variable([[3.,4]])
y=tf.matmul(w1,[[9.],[10]])
grads=tf.gradients(y,[w1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gradval=sess.run(grads)
    print(gradval)
 
import tensorflow as tf
w1 = tf.Variable([[1.,2]])
w2 = tf.Variable([[3.,4]])

y = tf.matmul(w1, [[9.],[10]])
#grads = tf.gradients(y,[w1,w2])#w2不相干，会报错
grads = tf.gradients(y,[w1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gradval = sess.run(grads)
    print(gradval)
    
    
    
    
import tensorflow as tf

tf.reset_default_graph()
w1=tf.get_variable('w1',shape=[2])
w2=tf.get_variable('w2',shape=[2])
w3=tf.get_variable('w3',shape=[2])
w4=tf.get_variable('w4',shape=[2])

y1=w1+w2+w3
y2=w3+w4

##grad_ys求梯度的输入值
gradients=tf.gradients([y1,y2],[w1,w2,w3,w4],grad_ys=[tf.convert_to_tensor([1.,2.]),tf.convert_to_tensor([3.,4.])])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(gradients))
    
    