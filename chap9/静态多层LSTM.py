# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 19:21:32 2019

@author: HAX
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/", one_hot=True)

n_input = 28 # MNIST data 输入 (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10  # MNIST 列别 (0-9 ，一共10类)
batch_size = 128


##初始化
tf.reset_default_graph()
# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

stack_rnn=[]
for i in range(3):
    stack_rnn.append(tf.contrib.rnn.LSTMCell(n_hidden))
mcell=tf.contrib.rnn.MultiRNNCell(stack_rnn)

x1=tf.unstack(x,n_steps,1)
outputs,states=tf.contrib.rnn.static_rnn(mcell,x1,dtype=tf.float32)
print('type(outputs)',type(outputs))
print('outputs[-1].shape',outputs[-1].shape)
print(outputs)
print(outputs[-1])##最后一个时间步对应的输出
pred=tf.contrib.layers.fully_connected(outputs[-1],n_classes,activation_fn=None)
##超参
learning_rate=0.001
training_iters=100000
display_step=10
total_batch=int(training_iters/batch_size)

##定义损失和梯度下降
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

##预测准确率
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if i% display_step == 0:
            # 计算批次数据的准确率
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print ("Iter " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
    print (" Finished!")
     
    # 计算准确率 for 128 mnist test images
    test_len = 100
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print ("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))