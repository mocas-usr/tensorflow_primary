# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 20:20:19 2019

@author: HAX
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
##提取mnist
mnist=input_data.read_data_sets('/data/',one_hot=True)


##初始化网络参数
n_steps=28
n_input=28
n_hidden=128
n_classes=10
tf.reset_default_graph()
##初始化变量
x=tf.placeholder('float',[None,n_steps,n_input])
y=tf.placeholder('float',[None,n_classes])


x1=tf.unstack(x,n_steps,1)
##构建rnn结构
stacked_rnn=[]
stacked_bw_rnn=[]
for i in range(3):
    stacked_rnn.append(tf.contrib.rnn.LSTMCell(n_hidden))
    stacked_bw_rnn.append(tf.contrib.rnn.LSTMCell(n_hidden))
    
outputs,_,_=rnn.stack_bidirectional_rnn(stacked_rnn,stacked_bw_rnn,x1,dtype=tf.float32)

pred = tf.contrib.layers.fully_connected(outputs[-1],n_classes,activation_fn = None)

##超参
training_iters=100000
batch_size=128
display_step=10
learning_rate=0.001
total_batch=int(training_iters/batch_size)
##损失和梯度下降
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

##计算准确率
acc_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(acc_pred,tf.float32))

##启动训练

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(total_batch):
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        batch_x=batch_x.reshape(batch_size,n_steps,n_input)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if i%display_step==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            loss=sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            print('iter'+str(i)+',   minibatch loss='+'{:.6f}'.format(loss)+
                  '    accuracy = '+'{:.5f}'.format(acc))
            
    print('--------finish-----')
    ##测试
    test_len=128
    test_data=mnist.test.images[:128].reshape((-1,n_steps,n_input))
    test_label=mnist.test.labels[:128]
    print('testing accuracy',sess.run(accuracy,feed_dict={x:test_data,y:test_label}))
            
        



