


import  cifar10_input
import tensorflow as tf
import pylab 
import numpy as np

#取数据
batch_size = 12
data_dir = '/chap8/cifar10data_bin'
images_test, labels_test = cifar10_input.inputs(eval_data = True, batch_size = batch_size)


sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
tf.train.start_queue_runners(sess=sess)
image_batch, label_batch = sess.run([images_test, labels_test])
print("__\n",image_batch[0])

print("__\n",label_batch[0])
pylab.imshow(  (image_batch[0]-np.min(image_batch[0]))  / (np.max(image_batch[0])-np.min(image_batch[0]) )   )
pylab.show()


import numpy as np
from scipy.misc import imsave 
import pylab

filename='./cifar10data_bin/test_batch.bin'
bytestream=open(filename,'rb')
buf=bytestream.read(10000*(1+32*32*3))
bytestream.close()

data=np.frombuffer(buf,dtype=np.uint8)
data=data.reshape(10000,1+32*32*3)
labels_images=np.hsplit(data,[1])
labels=labels_images[0].reshape(10000)
images=labels_images[1].reshape(10000,32,32,3)

print('images.shape',images.shape)
images=np.reshape(images,(10000,3,32,32))
images=images.transpose(0,2,3,1)
img=images[0]

#img0=np.reshape(images[0],(3,32,32))
print(labels[0])
#pylab.imshow(img0)
pylab.imshow(img)
pylab.show()
def input_cifar():
    filename='./cifar10data_bin/test_batch.bin'
    bytestream=open(filename,'rb')
    buf=bytestream.read(10000*(1+32*32*3))
    bytestream.close()

    data=np.frombuffer(buf,dtype=np.uint8)
    data=data.reshape(10000,1+32*32*3)
    labels_images=np.hsplit(data,[1])
    labels=labels_images[0].reshape(10000)
    images=labels_images[1].reshape(10000,32,32,3)
    images=np.reshape(images,(10000,3,32,32))
    images=images.transpose(0,2,3,1)
    print('-----------input cifar over----------')