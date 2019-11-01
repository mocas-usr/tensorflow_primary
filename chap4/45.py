# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:59:46 2019

@author: HAX
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


plotdata = { "batchsize":[], "loss":[] }
def moving_average(a, w=10):
    if len(a) < w: 
        return a[:]    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


#生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声
#图形显示
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

tf.reset_default_graph()##全部释放资源图

##定义ip和端口
strps_hosts='localhost:1681'
strworker_hosts='localhost:1682,loaclhost:1683'
##定义角色名称
strjob_name='ps'
task_index=0
#将字符串转换成数组
ps_hosts=strps_hosts.split(',')
worker_hosts=strworker_hosts.split(',')
print(ps_hosts)
cluster_spec=tf.train.ClusterSpec({'ps':ps_hosts,'worker':worker_hosts})

##创建server
server=tf.train.Server({'ps':ps_hosts,'worker':worker_hosts},job_name=strjob_name,task_index=task_index)
##ps使用join进行等待
if strjob_name=='ps':
    print('wait')
    server.join()
    
    
    
with tf.device(tf.train.replica_device_setter(worker_device='/job'))