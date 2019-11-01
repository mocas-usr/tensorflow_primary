# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:32:05 2019

@author: HAX
"""

from tensorflow.examples.tutorials.mnist import input_data
import pylab
import tensorflow as tf

##导入mnist数据集
mnist=input_data.read_data_sets('MNIST/data/')

print('输入数据',mnist.train.images)
print('输入数据的shape',mnist.train.images.shape)


im=mnist.train.images[1]
im=im.reshape(-1,28)
pylab.imshow(im)
pylab.show()

print('测试数据：',mnist.test.images.shape)
print('测试数据shape',mnist.validation.images.shape)

def maxout(inputs,num_units,axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0]=-1
    if axis is None:
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs

#图初始化
tf.reset_default_graph()
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.int32,[None])

w=tf.Variable(tf.random_normal([784,100]))
b=tf.Variable(tf.zeros([100]))

z=tf.matmul(x,w)+b

z=maxout(z,50)

w2=tf.Variable(tf.random_normal([50,10]))
b2=tf.Variable(tf.zeros([10]))

pred=tf.matmul(z,w2)+b2

##参数初始化
learning_rate=0.04
training_epochs=20
batch_size=20
display_step=1
training_size=mnist.train.num_examples
##损失和梯度下降
cost=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=pred))
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


##开始训练
with tf.Session() as sess:
    #初始化变量
    sess.run(tf.global_variables_initializer())
    ##循环训练
    for epoch in range(training_epochs):
        ##batch训练
        avg_cost=0.
        total_batch=int(training_size/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            #运行损失和梯度
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            
            avg_cost+=c/total_batch
            #显示训练信息
        if ((epoch+1)%display_step==0):
                print('epoch:','%04d'%(epoch+1),'cost=','{:.9f}'.format(avg_cost))
                
    print('\n','--------finish-------')
                
                
            
