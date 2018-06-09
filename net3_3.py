#-*-coding:UTF-8-*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np 
from datetime import datetime 
import math
import time
from create_input1 import read_and_decode
#data_in输入图片
data_in=tf.placeholder(tf.float32,[None,256,256,1])
#dropout层的数据保留百分比
keep_prob=tf.placeholder(tf.float32)
is_test=tf.placeholder(tf.bool)
def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
    n_in=input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(scope+"w",shape=[kh,kw,n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        #kernel=tf.Variable(tf.truncated_normal([kh,kw,n_in,n_out],stddev=0.1),trainable=True,name='w') 
        conv=tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding='SAME')
        n_out_conv=conv.get_shape()[-1].value
        bias_init_val=tf.constant(0.0,shape=[n_out_conv],dtype=tf.float32)
        #offset_init_val=tf.constant(0.0,shape=[n_out_conv],dtype=tf.float32)
        #scale_init_val=tf.constant(1.0,shape=[n_out_conv],dtype=tf.float32)
        #offset=tf.Variable(offset_init_val,trainable=True,name='offset')
        biases=tf.Variable(bias_init_val,trainable=True,name='b')
        z=tf.nn.bias_add(conv,biases)
        #scale=tf.Variable(scale_init_val,trainable=True,name='scale')
        #z=tf.nn.bias_add(conv,biases)
        #exp_moving_avg = tf.train.ExponentialMovingAverage(0.998)
        #bnepsilon = 1e-5
        #mean, variance = tf.nn.moments(conv, [0])
        #update_moving_averages = exp_moving_avg.apply([mean, variance])
        #m = tf.cond(test, lambda: exp_moving_avg.average(mean), lambda: mean)
        #v = tf.cond(test, lambda: exp_moving_avg.average(variance), lambda: variance)
        #z = tf.nn.batch_normalization(conv, m, v, offset, scale, bnepsilon)
        activation=tf.nn.relu(z,name=scope)
        p+=[kernel,biases]
        
        return activation

#全链接层
def fc_op(input_op,name,n_out,p):
    n_in=input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(scope+"w",shape=[n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases=tf.Variable(tf.constant(0.0,shape=[n_out],dtype=tf.float32),trainable=True,name='b')
        activation=tf.nn.relu_layer(input_op,kernel,biases,name=scope)
        #activation=tf.matmul(input_op,kernel)+biases
        p+=[kernel,biases]

        return activation
#max-pooling最大池化层
def mpool_op(input_op,name,kh,kw,dh,dw):
    return tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding='SAME',name=name)
def apool_op(input_op,name,kh,kw,dh,dw):
    return tf.nn.avg_pool(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding='SAME',name=name)
p=[]
#具体网络结构(改）

#conv1_1=conv_op(data_in,name='conv1_1',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
#conv1_2=conv_op(conv1_1,name='conv1_2',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
conv1_1=conv_op(data_in,name='conv1_1',kh=7,kw=7,n_out=64,dh=2,dw=2,p=p)
#pool1=mpool_op(conv1_1,name='pool1',kh=2,kw=2,dw=2,dh=2) 

#conv2_1=conv_op(pool1,name='conv2_1',kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)
#conv2_2=conv_op(conv2_1,name='conv2_2',kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)
conv2_1=conv_op(conv1_1,name='conv2_1',kh=7,kw=7,n_out=128,dh=2,dw=2,p=p)
#pool2=mpool_op(conv2_1,name='pool2',kh=2,kw=2,dw=2,dh=2)

#conv3_1=conv_op(pool2,name='conv3_1',kh=3,kw=3,n_out=256,dh=1,dw=1,p=p)
#conv3_2=conv_op(conv3_1,name='conv3_2',kh=3,kw=3,n_out=256,dh=1,dw=1,p=p)
#conv3_3=conv_op(conv3_2,name='conv2_3',kh=3,kw=3,n_out=256,dh=1,dw=1,p=p)
conv3_1=conv_op(conv2_1,name='conv3_1',kh=7,kw=7,n_out=256,dh=1,dw=1,p=p)
#pool3=mpool_op(conv2_1,name='pool3',kh=2,kw=2,dw=1,dh=1)


#conv4_1=conv_op(pool3,name='conv4_1',kh=3,kw=3,n_out=256,dh=1,dw=1,p=p)
#conv4_2=conv_op(conv4_1,name='conv4_2',kh=3,kw=3,n_out=256,dh=1,dw=1,p=p)
#conv4_3=conv_op(conv4_2,name='conv4_3',kh=3,kw=3,n_out=256,dh=1,dw=1,p=p)
conv4_1=conv_op(conv3_1,name='conv4_1',kh=7,kw=7,n_out=512,dh=1,dw=1,p=p)
#pool4=mpool_op(conv4_1,name='pool4',kh=2,kw=2,dw=1,dh=1)

#conv5_1=conv_op(pool4,name='conv5_1',kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)
#conv5_2=conv_op(conv5_1,name='conv5_2',kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)
#conv5_3=conv_op(conv5_2,name='conv5_3',kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)
conv5_1=conv_op(conv4_1,name='conv5_1',kh=7,kw=7,n_out=512,dh=1,dw=1,p=p)
pool5=mpool_op(conv5_1,name='pool5',kh=2,kw=2,dw=1,dh=1)
"""
#conv6_1=conv_op(pool5,name='conv6_1',kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)
#conv6_2=conv_op(conv6_1,name='conv6_1',kh=1,kw=1,n_out=512,dh=1,dw=1,p=p)
#conv6_3=conv_op(conv6_2,name='conv6_3',kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)
conv6_1=conv_op(pool5,name='conv6_1',kh=7,kw=7,n_out=512,dh=1,dw=1,p=p)
pool6=mpool_op(conv6_1,name='pool6',kh=2,kw=2,dw=2,dh=2)

conv7_1=conv_op(pool6,name='conv7_1',kh=7,kw=7,n_out=512,dh=1,dw=1,p=p)
#conv7_2=conv_op(conv7_1,name='conv7_2',kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)
pool7=mpool_op(conv7_1,name='pool7',kh=2,kw=2,dw=2,dh=2)

conv8_1=conv_op(pool7,name='conv8_1',kh=7,kw=7,n_out=768,dh=1,dw=1,p=p)
#conv8_2=conv_op(conv8_1,name='conv8_2',kh=3,kw=3,n_out=768,dh=1,dw=1,p=p)
pool8=mpool_op(conv8_1,name='pool8',kh=2,kw=2,dw=2,dh=2)

conv9_1=conv_op(pool8,name='conv9_1',kh=7,kw=7,n_out=768,dh=1,dw=1,p=p)
#conv9_2=conv_op(conv9_1,name='conv9_2',kh=1,kw=1,n_out=768,dh=1,dw=1,p=p)
#conv9_3=conv_op(conv9_2,name='conv9_3',kh=3,kw=3,n_out=768,dh=1,dw=1,p=p)
pool9=mpool_op(conv9_1,name='pool9',kh=2,kw=2,dw=2,dh=2)


conv10_1=conv_op(pool9,name='conv10_1',kh=7,kw=7,n_out=768,dh=1,dw=1,p=p)
pool10=mpool_op(conv10_1,name='pool10',kh=2,kw=2,dw=2,dh=2)
"""
shp=pool5.get_shape()
flattened_shape=shp[1].value*shp[2].value*shp[3].value
resh1=tf.reshape(pool5,[-1,flattened_shape],name='resh1')
            
fc6=fc_op(resh1,name='fc6',n_out=768,p=p)
fc6_drop=tf.nn.dropout(fc6,keep_prob,name='fc6_drop')

fc7=fc_op(fc6_drop,name='fc7',n_out=512,p=p)
fc7_drop=tf.nn.dropout(fc7,keep_prob,name='fc7_drop')
            
fc8=fc_op(fc7_drop,name='fc8',n_out=336,p=p)
fc8=tf.nn.softmax(fc8)
#values,indices=tf.nn.top_k(fc8,5)
#正确的数据标签
label=tf.placeholder(tf.float32,[None,336])
#计算准确率BP算法（后向传播)
cross_entropy=tf.reduce_mean(-tf.reduce_sum(label*tf.log(tf.clip_by_value(fc8,1e-10,1.0))))
#cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc8,labels=label)

#correct_label=tf.reshape(tf.argmax(label,1),[16,1])
#correct_prediction=tf.cast(tf.equal(tf.cast(indices,tf.int64),correct_label),tf.int32)
#correct_prediction=tf.reduce_sum(correct_prediction,reduction_indices=[1])
#correct_prediction=tf.reduce_sum(correct_prediction,reduction_indices=[0])

#accuracy=tf.cast(correct_prediction,tf.float32)/16
##
correct_prediction=tf.equal(tf.argmax(fc8,1),tf.argmax(label,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#梯度下降算法
train_step=tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(cross_entropy)
#train_step=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
#开始训练
train_img,train_label=read_and_decode("train_per.tfrecords")
test_img,test_label=read_and_decode("test_per.tfrecords")
train_img_batch,train_label_batch=tf.train.shuffle_batch([train_img,train_label],batch_size=128,capacity=2000,min_after_dequeue=1)

test_img_batch,test_label_batch=tf.train.shuffle_batch([test_img,test_label],batch_size=16,capacity=200,min_after_dequeue=1)
avg=0;
for_test,_=tf.nn.top_k(data_in,1)
init=tf.global_variables_initializer()
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)
    try:
        for i in range(2000):
            train_imgs,train_labels=sess.run([train_img_batch,train_label_batch])
            test_imgs,test_labels=sess.run([test_img_batch,test_label_batch])
            test_labels=tf.one_hot(test_labels,336,dtype=tf.float32).eval()
            train_labels=tf.one_hot(train_labels,336,dtype=tf.float32).eval()
            train_imgs=np.array(train_imgs)
            test_imgs=np.array(test_imgs) 
            #kernel=p[0]
            #print(kernel.eval())
            train_step.run({data_in:train_imgs,label:train_labels,keep_prob:1.0}) 
            if i%10==0:
                print(sess.run(cross_entropy,feed_dict={data_in:test_imgs,label:test_labels,keep_prob:1.0}))
                #print(sess.run(cross_entropy,feed_dict={data_in:train_imgs,label:train_labels,keep_prob:0.9}))
                acc=accuracy.eval({data_in:test_imgs,label:test_labels,keep_prob:1.0})
                #print(accuracy.eval({data_in:test_imgs,label:test_labels,keep_prob:1.0}))
                print(acc)
                if i > 100:
                    avg=avg+acc
                    print('the total avg: ',avg/((i/10)-10),i)
        saver.save(sess,'./model/model.ckpt')
    #except tf.errors.OpError:
        #print('Oops,Errors!')
    finally:
        coord.request_stop()
    coord.join(threads)
