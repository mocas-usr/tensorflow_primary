# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:23:31 2019

@author: HAX
"""

import tensorflow as tf
from tensorflow.contrib import rnn

# 导入 MINST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/", one_hot=True)

##初始化网络参数
n_input=28
n_steps=28
n_hidden=128
n_classes=10

#初始化图
tf.reset_default_graph()

##初始化变量
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
x1 = tf.unstack(x, n_steps, 1)
lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
#反向cell
lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,x,
                                              dtype=tf.float32)
print(len(outputs),outputs[0].shape,outputs[1].shape)
outputs = tf.concat(outputs, 2)
outputs = tf.transpose(outputs, [1, 0, 2])

pred = tf.contrib.layers.fully_connected(outputs[-1],n_classes,activation_fn = None)

#参数设置
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10
total_batch=int(training_iters/batch_size)

##损失和梯度下降
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
##准确率计算
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(total_batch):
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if i%display_step==0:
            loss=sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            print ("Iter " + str(i) + ", Minibatch Loss= " +
                   "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
    print (" Finished!")

    # 计算准确率 for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print ("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))


