# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:35:23 2019

@author: HAX
"""

import tensorflow as tf
hello=tf.constant('hello,world,tensorflow')
sess=tf.Session()
print(sess.run(hello))
sess.close()


import tensorflow as tf
a=tf.constant(20)
b=tf.constant(40)
with tf.Session() as sess:
     print('相加%i'%sess.run(a+b))
     print('相乘%i'%sess.run(a*b))
     


import tensorflow as tf
a=tf.placeholder(tf.int16)
b=tf.placeholder(tf.int16)    
add=tf.add(a,b)
mul=tf.multiply(a,b)

with tf.Session() as sess:
    print('相加之和',sess.run(add,feed_dict={a:3,b:4}))
    print('相乘之和',sess.run(mul,feed_dict={a:3,b:4}))
    print(sess.run([add,mul],feed_dict={a:3,b:4}))
 
    
    ##保存模型
##之前是对模型graph的操作
saver=tf.train.Saver()#生成saver
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess,'sever')
    
    
##载入模型
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,'saver')
    
    
    