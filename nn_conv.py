#!/usr/bin/python
#-*-coding:UTF-8-*-

import tensorflow as tf
from python_speech_features import mfcc,delta
import numpy as np
import scipy.io.wavfile as wav
from read_cep import read_cep
import os 
import sklearn.preprocessing

data_in=tf.placeholder(tf.float32,[None,1500,13])
label=tf.placeholder(tf.float32,[None,63])
data_img=tf.reshape(data_in,[-1,1500,13,1])
out_units=63
labels=[]
audios=[]
label_binarizer=""

def find_max_hl(voices):
    h,l=0,0
    for voice in voices:
        a,b=np.array(voice).shape
        if a>h:
            h=a
        if b>l:
            l=b
    #return h,l
    return 1500,13
def construct_input(voices):
    h,l=find_max_hl(voices)
    new_audios=[]
    for voice in voices:
        zero_matrix=np.zeros([h,l],np.float32)
        a,b=np.array(voice).shape
        for i in range(a):
            for j in range(b):
                zero_matrix[i,j]=zero_matrix[i,j]+voice[i,j]
        new_audios.append(zero_matrix)
    return new_audios,h,l

def read_wav_mfcc(wav_path):
    map_path,map_relative=[str(wav_path)+str(x) for x in os.listdir(wav_path) if os.path.isfile(str(wav_path)+str(x))],[y for y in os.listdir(wav_path)]
    for idx,file_name in enumerate(map_path):
        fs,audio=wav.read(file_name)
        a=mfcc(audio,samplerate=fs,nfft=512)
        audios.append(a)
        labels.append(int((map_relative[idx].split(".")[0])[:-1]))
    new_voices,h,l=construct_input(audios)
    if label_binarizer == "":
        binarizer = sklearn.preprocessing.LabelBinarizer()
    else:
        binarizer=label_binarizer
    binarizer.fit(range(max(labels)+1))
    new_ls=binarizer.transform(labels)
    return new_voices,new_ls
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_op(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.01)#生成一个截断的正态分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01,shape=shape)
    return tf.Variable(initial)

train_feature_batch,train_label_batch=read_wav_mfcc("/home/liuchen/mfcc/MFCC/voice/inputs/train/")
test_feature_batch,test_label_batch=read_wav_mfcc("/home/liuchen/mfcc/MFCC/voice/inputs/test/")


W_conv1=weight_variable([5,5,1,32])
W_conv2=weight_variable([5,5,32,64])

b_conv1=bias_variable([32])
b_conv2=bias_variable([64])

conv1=conv2d(data_img,W_conv1)+b_conv1
pool1=max_op(conv1)

shp=pool1.get_shape()
flattened_shape=shp[1].value*shp[2].value*shp[3].value
resh1=tf.reshape(pool1,[-1,flattened_shape])

W_fc1=weight_variable([flattened_shape,out_units])

b_fc1=bias_variable([out_units])

fc1=tf.matmul(resh1,W_fc1)+b_fc1
fc1=tf.nn.sigmoid(fc1)
fc1=tf.nn.softmax(fc1)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(label*tf.log(tf.clip_by_value(fc1,1e-10,1.0)),reduction_indices=[1]))

correct_prediction=tf.equal(tf.argmax(fc1,1),tf.argmax(label,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#train_step=tf.train.GradientDescentOptimizer(1e2).minimize(cross_entropy)
train_step=tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

init=tf.global_variables_initializer()



with tf.Session() as sess:
     sess.run(init)
     coord=tf.train.Coordinator()
     threads=tf.train.start_queue_runners(coord=coord)
     train_exp,train_l=read_wav_mfcc("/home/liuchen/mfcc/MFCC/voice/inputs/train/")
     test_exp,test_l=read_wav_mfcc("/home/liuchen/mfcc/MFCC/voice/inputs/test/")
     try:
         for i in range(5000):
             #train_exp,train_l=sess.run([train_feature_batch,train_label_batch])
             #test_exp,test_l=sess.run([test_feature_batch,test_label_batch])
             #train_exp=tf.convert_to_tensor(train_exp)
             #test_exp=tf.convert_to_tensor(test_exp)
             train_step.run({data_in:train_exp,label:train_l})
             if i%10 ==0:
                 print("train: ",sess.run(cross_entropy,feed_dict={data_in:train_exp,label:train_l}))
                 print("test: ",sess.run(cross_entropy,feed_dict={data_in:test_exp,label:test_l}))
                 
                 #print(i,"train",accuracy.eval({data_in:train_exp,label:train_l}))
                 print(i,"test",accuracy.eval({data_in:test_exp,label:test_l}))
     finally:
         coord.request_stop()
     coord.join(threads)
     sess.close()        
