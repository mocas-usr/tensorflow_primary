# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:50:25 2019

@author: HAX
"""

import tensorflow as tf
##导入mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('/data/',one_hot=True)

n_input=28##mnist输入（image shape 28*28）
n_step=28   ##序列个数
n_hidden=128    ##隐藏层个数
n_classes=10##MNIST的分类个数
tf.reset_default_graph()
##定义变量，占位符
x=tf.placeholder('float',[None,n_step,n_input])
y=tf.placeholder('float',[None,n_classes])

x1=tf.unstack(x,n_step,1)
lstm_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
outputs,states=tf.contrib.rnn.static_rnn(lstm_cell,x1,dtype=tf.float32)
#print('output.shape',outputs.shape)
print('output[-1].shape',outputs[-1].shape)
pred=tf.contrib.layers.fully_connected(outputs[-1],n_classes,activation_fn=None)



##初始化参数
learning_rate=0.001
training_iters=100000
batch_size = 128
display_step = 10
total_batch=int(training_iters/batch_size)
#损失和梯度下降计算
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

##准确率计算
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


#启动循环训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step=1
    for i in range(total_batch):
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        batch_x=batch_x.reshape((batch_size,n_step,n_input))
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if i%display_step==0:
            ##计算批次准确率
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print ("Iter " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
    print('---------finish----------')
    
    
    ##计算准确率
    test_len=128
    test_data=mnist.test.images[:128].reshape((-1,n_step,n_input))
    test_label = mnist.test.labels[:test_len]
    print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))    
    
        
    
    
