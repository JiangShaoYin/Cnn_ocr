#	  coding:utf-8
#	 @file    forward.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-05 21:35:21
 
 
import tensorflow as tf
import numpy as np


IMAGE_HEIGHT = 32					#图像的像素尺寸
IMAGE_WIDTH = 256					#图像的像素尺寸
NUM_CHANNELS = 1
CONV1_SIZE = 5          #第1层卷积的核长
CONV1_KERNEL_NUM = 32   #第1层卷积层的深度（核数）
CONV2_SIZE =5						#第2层卷积的核长
CONV2_KERNEL_NUM = 64   #第2层卷积层的深度（核数）
FC_SIZE = 512           #全连接层的神经元个数
OUTPUT_NODE = 180       #全连接第2层的神经元个数
INPUT_NODE = 32*256


def get_weight(shape, regularizer):#第1层卷积的核长
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))	#生成shape形状的参数矩阵w，其值服从平均值和标准偏差的正态分布，如果生成的值大于平均值2倍标准偏差，丢弃该值并重新选择。
    if regularizer!= None:																	#如果启用正则化
        tf.add_to_collection('losses',											#用l2正则化w，并将结果加入losses集合
                            tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w 

def get_bais(shape):
    b = tf.Variable(tf.zeros(shape))												#生成全shape形状的全0矩阵
    return b

def conv2d(x, w):
    return tf.nn.conv2d(x,
            w,
            strides = [1,1,1,1], 
            padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def forward(x, train, regularizer):
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS ,CONV1_KERNEL_NUM], regularizer)#第1个卷积层，kernel_num是卷积层的核数,卷积层1的参数矩阵conv1_w==5*5*1*32
    conv1_b = get_bais([CONV1_KERNEL_NUM])  							#CONV1_KERNEL_NUM==32，卷积层的偏执项，即卷积层的核数,
    conv1 = conv2d(x, conv1_w)              							# x的尺寸为100*28*28*1（100张图），conv1_w==5*5*1*32，计算结果尺寸，32个28*28的矩阵（每个图片对应1个28*28*32的计算结果）
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))	  #将卷积层1的计算结果，非线性激活
    pool1 = max_pool_2x2(relu1)														#2*2池化，输出14*14*32（32个14*14）的计算结果

    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)#第2个卷积层，conv2_w==5*5*1*64
    conv2_b = get_bais([CONV2_KERNEL_NUM]) 							 #CONV1_KERNEL_NUM==64，卷积层的偏执项，即卷积层的核数,
    conv2 = conv2d(pool1, conv2_w)             					 #pool1的尺寸100*14*14*32，conv2_w==5*5*1*64，算结果尺寸，64个14*14的矩阵（每个图片对应1个14*14*64的计算结果）
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))	 #将卷积层1的计算结果，非线性激活
    pool2 = max_pool_2x2(relu2)													 #2*2池化，输出7*7*64（64个14*14）的计算结果

    pool_shape = pool2.get_shape().as_list()       						#输出矩阵pool2的维度，存入list中（pool_shape)
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]			#nodes==7*7*64==3136，特征点的长度*宽度*深度==特征点的个数
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes]) 			#pool_shape[0]==100，（一次喂入的图片个数==batch），，将pool2拉成一个batch行，每行nodes个点的二维数组
																															#reshaped是100*3136的矩阵，每1行代表1个图片
																															
    fc1_w = get_weight([nodes, FC_SIZE], regularizer)         #全连接层参数fc1_w 为3136*512的矩阵
    fc1_b = get_bais([FC_SIZE])																#偏执项个数512
    fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_w) + fc1_b)				#fc1的计算结果为100*512的矩阵，代表每个图片对应的512个神经元计算值（100张图片）

    if train:													#如果是训练阶段，则将上一轮的输出fc1，随机舍去一定比例的计算结果（神经元）
        fc1 = tf.nn.dropout(fc1, 0.5)

    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)		#全连接层参数 fc2_w 是 512*10 的矩阵
    fc2_b = get_bais([OUTPUT_NODE])														#偏执项个数10
    y = tf.matmul(fc1, fc2_w) + fc2_b                         #fc1==100*512，fc2_w==512*10， 计算结果,100*10的矩阵（100张图片）
    return y
