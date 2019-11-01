# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 20:00:21 2019

@author: HAX
"""

import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import tensorflow as tf

##载入图片，并显示
mying=mping.imread('img.jpg')##显示图片
plt.imshow(mying)##显示图片
plt.axis('off')
plt.show()
print(mying.shape)


##定义占位符，卷积
full=np.reshape(mying,[1,200,280,3])
input_full=tf.placeholder(tf.float32,shape=[1,200,280,3])
filter=tf.Variable(tf.constant([[-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0],[-2.0,-2.0,-2.0],
                                [0,0,0],[2.0,2.0,2.0],[-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0]],shape=[3,3,3,1]))

op=tf.nn.conv2d(input_full,filter,strides=[1,1,1,1],padding='SAME')
vop=tf.cast((op-tf.reduce_min(op))/(tf.reduce_max(op)-tf.reduce_min(op))*255,tf.uint8)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t,f=sess.run([vop,filter],feed_dict={input_full:full})
    print(t.shape)
    t=np.reshape(t,[200,280])
    plt.imshow(t,cmap='Greys_r')
    plt.axis('off')
    plt.show()

