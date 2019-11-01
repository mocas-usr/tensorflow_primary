# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:10:46 2019

@author: HAX
"""

import tensorflow as tf
from tensorflow.contrib import rnn

# 导入 MINST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/", one_hot=True)

#初始化网络参数
# Network Parameters
n_input = 28 # MNIST data 输入 (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10  # MNIST 列别 (0-9 ，一共10类)

#初始化图和变量
tf.reset_default_graph()

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
x1 = tf.unstack(x, n_steps, 1)
lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
# 反向cell
lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x1,
                                              dtype=tf.float32)

print(outputs[0].shape,len(outputs))
pred = tf.contrib.layers.fully_connected(outputs[-1],n_classes,activation_fn = None)
##超参数
#参数设置
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10
total_batch=int(training_iters/batch_size)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
##启动session训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(total_batch):
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        batch_x=batch_x.reshape(batch_size,n_steps, n_input)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if i%display_step==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            loss=sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            print ("Iter " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            
    print('------finish----------')
    # 计算准确率 for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print ("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

            

