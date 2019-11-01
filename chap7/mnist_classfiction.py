# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:00:30 2019

@author: HAX
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST/data/',one_hot=True)

print(mnist.test.images.shape)
##参数设置

learning_rate=0.001
training_epcohs=25
batch_size=100
display_step=1
training_size=mnist.train.num_examples
total_batch=int(training_size/batch_size)
#网络结构参数设置
n_hidden1=256
n_hidden2=256
n_input=784
n_class=10

##图初始化
x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_class])

##创建model
def multilayer_forward(x,weights,bias):
    layer_1=tf.add(tf.matmul(x,weights['h1']),bias['b1'])
    layer_1=tf.nn.relu(layer_1)
    
    layer_2=tf.add(tf.matmul(layer_1,weights['h2']),bias['b2'])
    layer_2=tf.nn.relu(layer_2)
    
    out_layer=tf.matmul(layer_2,weights['out'])+bias['out']
    return out_layer

weights={'h1':tf.Variable(tf.random_normal([n_input,n_hidden1])),
         'h2':tf.Variable(tf.random_normal([n_hidden1,n_hidden2])),
         'out':tf.Variable(tf.random_normal([n_hidden2,n_class]))}
bias={'b1':tf.Variable(tf.random_normal([n_hidden1])),
      'b2':tf.Variable(tf.random_normal([n_hidden2])),
      'out':tf.Variable(tf.random_normal([n_class]))  }

#输出值
pred=multilayer_forward(x,weights,bias)
##定义loss和优化器
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)


##运行训练

with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    
    ##训练循环
    for epoch in range(training_epcohs):
        avg_cost=0.
        for i in range(total_batch):
            ##batch训练
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            ##计算损失
            avg_cost+=c/total_batch
            
        ##显示训练信息
        if epoch%display_step==0:
            print('epoch:',(epoch+1),'cost:','{:.9f}'.format(avg_cost))
            
    print('------finish------')
    ##准确率计算
    correct=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accruacy=tf.reduce_mean(tf.cast(correct,'float'))
    print('accruacy:',accruacy.eval({x:mnist.test.images,y:mnist.test.labels}))
            
    

