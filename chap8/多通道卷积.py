# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:36:17 2019

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

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')  
                        
def avg_pool_6x6(x):
  return tf.nn.avg_pool(x, ksize=[1, 6, 6, 1],
                        strides=[1, 6, 6, 1], padding='SAME')
  
##变量初始化
x = tf.placeholder(tf.float32, [None, 24,24,3]) # cifar data image of shape 24*24*3
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 数字=> 10 classes

W_conv1 = weight_variable([5, 5, 3, 64])
b_conv1 = bias_variable([64])

x_image = tf.reshape(x, [-1,24,24,3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#######################################################多卷积核

W_conv2_5x5 = weight_variable([5, 5, 64, 64]) 
b_conv2_5x5 = bias_variable([64])

W_conv2_7x7 = weight_variable([7, 7, 64, 64]) 
b_conv2_7x7 = bias_variable([64]) 

W_conv2_3x3 = weight_variable([3, 3, 64, 64]) 
b_conv2_3x3 = bias_variable([64]) 

W_conv2_1x1 = weight_variable([1, 1, 64, 64]) 
b_conv2_1x1 = bias_variable([64]) 
##卷积各通道计算
h_conv2_1x1 = tf.nn.relu(conv2d(h_pool1, W_conv2_1x1) + b_conv2_1x1)
h_conv2_3x3 = tf.nn.relu(conv2d(h_pool1, W_conv2_3x3) + b_conv2_3x3)
h_conv2_5x5 = tf.nn.relu(conv2d(h_pool1, W_conv2_5x5) + b_conv2_5x5)
h_conv2_7x7 = tf.nn.relu(conv2d(h_pool1, W_conv2_7x7) + b_conv2_7x7)
print('h_conv2_7x7.shape',h_conv2_7x7.shape)
##第四个维度进行融合
h_conv2 = tf.concat([h_conv2_5x5,h_conv2_7x7,h_conv2_3x3,h_conv2_1x1],3)
#第二层池化
print('h_conv2.shape',h_conv2.shape)
h_pool2 = max_pool_2x2(h_conv2)
print('h_pool2.shape',h_pool2.shape)
W_conv3 = weight_variable([5, 5, 256, 10])
b_conv3 = bias_variable([10])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
print('h_conv3.shape',h_conv3.shape)

nt_hpool3=avg_pool_6x6(h_conv3)#10
print('nt_hpool3.shape',nt_hpool3.shape)
nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
print('nt_hool3flat.shape',nt_hpool3_flat.shape)
y_conv=tf.nn.softmax(nt_hpool3_flat)

##梯度下降优化
cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))
#
#不同的优化方法测测效果
#train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
#train_step = tf.train.AdagradOptimizer(1e-5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
##损失计算
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

for i in range(1500):#20000
  image_batch, label_batch = sess.run([images_train, labels_train])
  label_b = np.eye(10,dtype=float)[label_batch] #one hot
  
  train_step.run(feed_dict={x:image_batch, y: label_b},session=sess)
  
  if i%20 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:image_batch, y: label_b},session=sess)
    print( "step %d, training accuracy %g"%(i, train_accuracy))


image_batch, label_batch = sess.run([images_test, labels_test])
label_b = np.eye(10,dtype=float)[label_batch]#one hot
print ("finished！ test accuracy %g"%accuracy.eval(feed_dict={
     x:image_batch, y: label_b},session=sess))

