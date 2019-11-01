# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:07:39 2019

@author: HAX
"""

from tensorflow.examples.tutorials.mnist import input_data
import pylab

#导入数据
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

print('输入数据',mnist.train.images)
print('输入数据打印shape',mnist.train.images.shape)

im=mnist.train.images[1]
im=im.reshape(-1,28)
pylab.imshow(im)
pylab.show()

print('输入数据打印shape',mnist.test.images.shape)
print('输入数据打印shape',mnist.validation.images.shape)


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pylab

#导入数据
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
tf.reset_default_graph()
X=tf.placeholder(tf.float32,[None,784])
Y=tf.placeholder(tf.float32,[None,10])
#初始权重
w=tf.Variable(tf.random_normal([784,10]))
b=tf.Variable(tf.zeros([10]))

pred=tf.nn.softmax(tf.matmul(X,w)+b)


##定义超参
learning_rate=0.01
training_epochs=25
training_size=mnist.train.num_examples
batch_size=100
display_step=1
##损失函数
cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred),reduction_indices=1))
#优化
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

##开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #启动循环，开始训练
    for epoch in range(training_epochs):
        avg_cost=0.0
        total_batch=int(training_size/batch_size)
        ##循环所有数据集
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            ##运行优化器
            _,c=sess.run([optimizer,cost],feed_dict={X:batch_xs,Y:batch_ys})
            ##计算平均loss值
            avg_cost+=c/total_batch
        ##显示训练信息
        if (epoch+1)%display_step==0:
            print('epoch:','%04d'%(epoch+1),'cost=','{:.9f}'.format(avg_cost))
    print('-----------finish-------------')
    ##测试model
    correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
    ##计算准确率
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print('accuracy:',accuracy.eval({X:mnist.test.images,Y:mnist.test.labels}))
    
    #保存模型
    saver=tf.train.Saver()
    model_path='log/926model.ckpt'
    save_path=saver.save(sess,model_path)
    print('model saved in file:%s'%save_path)

    
print('restarting  a new Session')
with tf.Session() as sess:
    ##初始化变量，可以不初始化，因为后面还要导入
    sess.run(tf.global_variables_initializer())
    
    ##导入模型参数
    saver.restore(sess,model_path)
    
    
    '''
    tf.equal()对比两值是否相等，输出true，false
    tf.argmax()输出最大值所在位置
    tf.reduce_mean()求数组平均值
    tensor.eval(feed={})与session.run()功能基本等价
    
    '''
    
    #测试moel
    correct_prediction2=tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
    ##计算准确率
    accuracy=tf.reduce_mean(tf.cast(correct_prediction2,tf.float32))
    print('accuracy2:',accuracy.eval({X:mnist.test.images,Y:mnist.test.labels}))
    
    output=tf.argmax(pred,1)
    batch_xs,batch_ys=mnist.train.next_batch(2)
    outputval,predv=sess.run([output,pred],feed_dict={X:batch_xs})
    print('outputval,predv',outputval,predv,batch_ys)
    
    im=batch_xs[0]
    print(im.shape)
    im=im.reshape(-1,28)
    print(im.shape)
    pylab.imshow(im)
    pylab.show()
    
    
          




