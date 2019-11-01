# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:00:38 2019

@author: HAX
"""

import numpy as np
import tensorflow as tf

# 导入 MINST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/", one_hot=True)


##设置参数
n_input = 28 # MNIST data 输入 (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10  # MNIST 列别 (0-9 ，一共10类)
batch_size = 128

##初始图和参数
tf.reset_default_graph()
# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

gru = tf.contrib.rnn.GRUCell(n_hidden*2)
lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden)
mcell = tf.contrib.rnn.MultiRNNCell([lstm_cell,gru])
outputs,states  = tf.nn.dynamic_rnn(mcell,x,dtype=tf.float32)#(?, 28, 256)
outputs = tf.transpose(outputs, [1, 0, 2])

pred = tf.contrib.layers.fully_connected(outputs[-1],n_classes,activation_fn = None)

##超参
learning_rate = 0.001
training_iters = 100000
display_step = 10
total_batch=int(training_iters/batch_size)
##损失计算，梯度下降
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
##准确率计算
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(total_batch):
        batch_x,batch_y=mnist.train.next_batch(batch_size)
         # reshape数据成28*28的图片
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if i%display_step==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            loss=sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            
            print('iter'+str(i)+ ", Minibatch Loss= "+ "{:.6f}".format(loss)+ 
                  ", Training Accuracy= " + "{:.5f}".format(acc))
    
    print('----------finish---------')
# 计算准确率 for 128 mnist test images
    test_len=100
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
    
    