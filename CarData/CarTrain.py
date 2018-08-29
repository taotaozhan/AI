# -*- coding: utf-8 -*-


import tensorflow as tf
import read
import numpy as np

#image1, label1 = ReadMyOwnData.read_and_decode("car_test.tfrecords")
batch_size=300
#test
def compute_accuracy(v_xs,v_ys):
 global prediction
 y_pre=sess.run(prediction,feed_dict={x_image:v_xs})
 #out
 correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
 #test
 accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
 result =sess.run(accuracy,feed_dict={x_image:v_xs,y_image:v_ys})
 return result

def weight_variable(shape):
 initial=tf.truncated_normal(shape,stddev=0.1)
 return tf.Variable(initial)
def bias_variable(shape):
 initial=tf.constant(0.1,shape=shape)
 return tf.Variable(initial) 
#juan ji
def conv2d(x,W):
 return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_4x4(x):
 return tf.nn.max_pool(x,[1,4,4,1],strides=[1,4,4,1],padding='SAME')


#xs=tf.placeholder(tf.float32,[None,128*128])
#ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)

#x_image=tf.reshape(xs,[-1,128,128,3])
x_image= tf.placeholder(tf.float32, [batch_size,128,128,3])

y_image= tf.placeholder(tf.float32, [batch_size,1])

#conv1 layer bias_variable
W_conv1=weight_variable([5,5,3,32])#5*5 sao miao 3:out 32:l
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#output 128*128*32
h_pool1=max_pool_4x4(h_conv1)#output 32*32*32

#conv2 layer
W_conv2=weight_variable([5,5,32,64])#5*5 sao miao 1:out 32:l
b_conv2=bias_variable([64])#gao
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#output 32*32*32
h_pool2=max_pool_4x4(h_conv2)#output 8*8*128

#funcle layer

reshape = tf.reshape(h_pool2,[batch_size, -1])

dim = reshape.get_shape()[1].value

W_fc1 = weight_variable([dim, 1024])

b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)
#
#W_fc1=weight_variable([8*8*128,1024])
#b_fc1=bias_variable([1024])
#h_pool2_flat=tf.reshape(h_pool2,[-1,8*8*128])
#h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#funcle layer
W_fc2 = weight_variable([1024,2])
b_fc2 = bias_variable([2])
#W_fc2=weight_variable([1024,10])
#b_fc2=bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#h_pool2_flat2=tf.reshape(h_pool2,[-1,2*2*128])
#out=tf.matmul(h_fc1_drop,W_fc2)+b_fc2
#prediction=tf.nn.softmax(out)
#add output
#prediction=add_layer(xs,128*128,activation_funcation=tf.nn.softmax)

#loss

cross_entropy = tf.reduce_mean(-tf.reduce_sum
              (y_image* tf.log(prediction), reduction_indices=[1]))
#you hua loss
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y_image,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#init = tf.initialize_all_variables()
#image, label = read.read_and_decode("./car_train.tfrecords")
#sess = tf.Session()
#sess.run(init)
#coord=tf.train.Coordinator()
#threads= tf.train.start_queue_runners(coord=coord)

image, label = read.read_and_decode("./car_train.tfrecords")
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
coord=tf.train.Coordinator()
threads= tf.train.start_queue_runners(coord=coord)

#example = np.zeros((batch_size,128,128,3))
#l = np.zeros((batch_size,1))
try:   
  # for i in range(20):    
   # for epoch in range(200):
      #example[epoch], l[epoch] = sess.run([image,label])
     # print(epoch)
   print(image)
   print(label)
   train_step.run(feed_dict={x_image: image, y_image: label, keep_prob: 0.5})        
   print(accuracy.eval(feed_dict={x_image: image, y_image: label, keep_prob: 0.5})) 
except tf.errors.OutOfRangeError:
        print('done!')
finally:
    coord.request_stop()
coord.join(threads)

tf.global_variables_initializer().run()


#init = tf.initialize_all_variables()

#with tf.Session() as sess:
 #sess.run(init)
 
 #image, label = read.read_and_decode("./car_train.tfrecords")
 #print(sess.run(train_step,feed_dict={x_image: image, y_image: label, keep_prob: 0.5}))
