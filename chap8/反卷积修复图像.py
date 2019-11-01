# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 19:17:47 2019

@author: HAX
"""

import tensorflow as tf
import numpy as np
import cifar10_input


#最大池化
def max_pool_with_argmax(net, stride):
    _, mask = tf.nn.max_pool_with_argmax( net,ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1],padding='SAME')
    mask = tf.stop_gradient(mask)
    net = tf.nn.max_pool(net, ksize=[1, stride, stride, 1],strides=[1, stride, stride, 1], padding='SAME') 
    return net, mask
#4*4----2*2--=2*2 【6，8，12，16】    
#反池化
def unpool(net, mask, stride):
    ksize = [1, stride, stride, 1]
    input_shape = net.get_shape().as_list()

    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range

    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret

##数据导入
batch_size=128
data_dir='./cifar10data_bin'
print('begin')
images_train, labels_train = cifar10_input.inputs2(eval_data = False,data_dir = data_dir, batch_size = batch_size)
images_test, labels_test = cifar10_input.inputs2(eval_data = True, data_dir = data_dir, batch_size = batch_size)
print("begin data")

##变量初始化
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
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
x=tf.placeholder(tf.float32,[batch_size,24,24,3])
y = tf.placeholder(tf.float32, [batch_size, 10]) # 0-9 数字=> 10 classes

w_conv1=weight_variable([5,5,3,64])
b_conv1=weight_variable([64])

x_image=tf.reshape(x,[-1,24,24,3])
h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1, mask1 = max_pool_with_argmax(h_conv1, 2)

w_conv2=weight_variable([5,5,64,64])
b_conv2=weight_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2,mask2=max_pool_with_argmax(h_conv2,2)
print(h_pool2.shape)

##开始反卷积，反池化
t_conv2 = unpool(h_pool2, mask2, 2)
t_pool1 = tf.nn.conv2d_transpose(t_conv2-b_conv2, w_conv2, h_pool1.shape,[1,1,1,1])#(128, 24, 24, 64)
print(t_conv2.shape,h_pool1.shape,t_pool1.shape)
t_conv1 = unpool(t_pool1, mask1, 2)
t_x_image = tf.nn.conv2d_transpose(t_conv1-b_conv1, w_conv1, x_image.shape,[1,1,1,1])


# 生成最终图像
#第一层卷积还原
t1_conv1 = unpool(h_pool1, mask1, 2)
t1_x_image = tf.nn.conv2d_transpose(t1_conv1-b_conv1, w_conv1, x_image.shape,[1,1,1,1])

# 生成最终图像
stitched_decodings = tf.concat((x_image, t1_x_image,t_x_image), axis=2)

decoding_summary_op = tf.summary.image('source/cifar', stitched_decodings)

w_conv3 = weight_variable([5, 5, 64, 10])
b_conv3 = bias_variable([10])
h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)

nt_hpool3=avg_pool_6x6(h_conv3)#10
nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
y_conv=tf.nn.softmax(nt_hpool3_flat)
##损失和梯度下降
cross_entropy = -tf.reduce_sum(y*tf.log(y_conv)) +(tf.nn.l2_loss(w_conv1)+tf.nn.l2_loss(w_conv2)+tf.nn.l2_loss(w_conv3))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
##预测准确率
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter('./log/', sess.graph)

tf.train.start_queue_runners(sess=sess)

for i in range(1500):#20000
  image_batch, label_batch = sess.run([images_train, labels_train])
  label_b = np.eye(10,dtype=float)[label_batch] #one hot
  
  train_step.run(feed_dict={x:image_batch, y: label_b},session=sess)
  #_, decoding_summary = sess.run([train_step, decoding_summary_op],feed_dict={x:image_batch, y: label_b})
  
  if i%20 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:image_batch, y: label_b},session=sess)
    print( "step %d, training accuracy %g"%(i, train_accuracy))
    print("cross_entropy",cross_entropy.eval(feed_dict={x:image_batch, y: label_b},session=sess))
    #summary_writer.add_summary(decoding_summary)
    
image_batch, label_batch = sess.run([images_test, labels_test])
label_b = np.eye(10,dtype=float)[label_batch]#one hot
print ("finished！ test accuracy %g"%accuracy.eval(feed_dict={
     x:image_batch, y: label_b},session=sess))
decoding_summary = sess.run(decoding_summary_op,feed_dict={x:image_batch, y: label_b})
summary_writer.add_summary(decoding_summary)


