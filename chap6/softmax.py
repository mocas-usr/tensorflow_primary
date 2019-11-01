# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:21:57 2019

@author: HAX
"""

import tensorflow as tf

labels=[[0,0,1],[0,1,0]]
logits=[[2,0.5,6],[0.1,0,3]]
logits_scaled=tf.nn.softmax(logits)
logits_scaled2=tf.nn.softmax(logits_scaled)

result1=tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
result2=tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits_scaled)
result3=-tf.reduce_sum(labels*tf.log(logits_scaled),1)

with tf.Session() as sess:
    print('scaled=',sess.run(logits_scaled))
    print('scaled2=',sess.run(logits_scaled2))
    
    print('rell=',sess.run(result1),'\n')
    
    print('rel2=',sess.run(result2),'\n')
    print('rel3=',sess.run(result3),'\n')
    

##标签总概率为1
labels=[[0.4,0.1,0.5],[0.3,0.6,0.1]]
result4=tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)

with tf.Session() as sess:
    print('rel4=',sess.run(result4),'\n')


#标签
labels=[2,1]
result5=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
with tf.Session() as sess:
    print('rel5；',sess.run(result5),'\n')


loss=tf.reduce_mean(result1)
with tf.Session() as sess:
    print('loss=',sess.run(loss))
    
