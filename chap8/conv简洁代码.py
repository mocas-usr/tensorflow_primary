# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:04:29 2019

@author: HAX
"""

import cifar10_input
import tensorflow as tf
import numpy as np

##数据导入
batch_size=128
data_dir='./cifar10data_bin'
print('begin')
images_train, labels_train = cifar10_input.inputs2(eval_data = False,data_dir = data_dir, batch_size = batch_size)
images_test, labels_test = cifar10_input.inputs2(eval_data = True, data_dir = data_dir, batch_size = batch_size)
print("begin data")

##变量初始化及输入
x=tf.placeholder(tf.float32,[None,24,24,3])
y=tf.placeholder(tf.float32,[None,10])

x_image=tf.reshape(x,[-1,24,24,3])

h_conv1=tf.contrib.layers.conv2d(x_image,64,[5,5],1,'SAME',activation_fn=tf.nn.relu)
h_pool1=tf.contrib.layers.max_pool2d(h_conv1,[2,2],stride=2,padding='SAME')

h_conv2=tf.contrib.layers.conv2d(h_pool1,64,[5,5],1,'SAME',activation_fn=tf.nn.relu)
h_pool2=tf.contrib.layers.max_pool2d(h_conv2,[2,2],stride=2,padding='SAME')

##全连接展开计算损失
nt_pool2=tf.contrib.layers.avg_pool2d(h_pool2,[6,6],stride=6,padding='SAME')
nt_pool2_flat=tf.reshape(nt_pool2,[-1,64])
y_conv=tf.contrib.layers.fully_connected(nt_pool2_flat,10,activation_fn=tf.nn.softmax)
##计算损失，梯度下降
cross_entropy=-tf.reduce_sum(y*tf.log(y_conv))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#计算准确率
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))
##运行计算
sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

for i in range(1500):
    image_batch, label_batch = sess.run([images_train, labels_train])
    label_b = np.eye(10,dtype=float)[label_batch] #one hot
    #train_step.run(feed_dict={x:image_batch, y: label_b},session=sess)
    sess.run(train_step,feed_dict={x:image_batch,y:label_b})
    if i%200 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:image_batch, y: label_b},session=sess)
        print( "step %d, training accuracy %g"%(i, train_accuracy))
    
image_batch, label_batch = sess.run([images_test, labels_test])
label_b = np.eye(10,dtype=float)[label_batch]#one hot
print ("finished！ test accuracy %g"%accuracy.eval(feed_dict={
     x:image_batch, y: label_b},session=sess))

