#	  coding:utf-8
#	 @file    forward.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-05 21:35:21
 
 
import tensorflow as tf
import numpy as np


IMAGE_HEIGHT = 32					#ͼ������سߴ�
IMAGE_WIDTH = 256					#ͼ������سߴ�
NUM_CHANNELS = 1
CONV1_SIZE = 5          #��1�����ĺ˳�
CONV1_KERNEL_NUM = 32   #��1���������ȣ�������
CONV2_SIZE =5						#��2�����ĺ˳�
CONV2_KERNEL_NUM = 64   #��2���������ȣ�������
FC_SIZE = 512           #ȫ���Ӳ����Ԫ����
OUTPUT_NODE = 180       #ȫ���ӵ�2�����Ԫ����
INPUT_NODE = 32*256


def get_weight(shape, regularizer):#��1�����ĺ˳�
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))	#����shape��״�Ĳ�������w����ֵ����ƽ��ֵ�ͱ�׼ƫ�����̬�ֲ���������ɵ�ֵ����ƽ��ֵ2����׼ƫ�������ֵ������ѡ��
    if regularizer!= None:																	#�����������
        tf.add_to_collection('losses',											#��l2����w�������������losses����
                            tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w 

def get_bais(shape):
    b = tf.Variable(tf.zeros(shape))												#����ȫshape��״��ȫ0����
    return b

def conv2d(x, w):
    return tf.nn.conv2d(x,
            w,
            strides = [1,1,1,1], 
            padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def forward(x, train, regularizer):
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS ,CONV1_KERNEL_NUM], regularizer)#��1������㣬kernel_num�Ǿ����ĺ���,�����1�Ĳ�������conv1_w==5*5*1*32
    conv1_b = get_bais([CONV1_KERNEL_NUM])  							#CONV1_KERNEL_NUM==32��������ƫִ��������ĺ���,
    conv1 = conv2d(x, conv1_w)              							# x�ĳߴ�Ϊ100*28*28*1��100��ͼ����conv1_w==5*5*1*32���������ߴ磬32��28*28�ľ���ÿ��ͼƬ��Ӧ1��28*28*32�ļ�������
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))	  #�������1�ļ������������Լ���
    pool1 = max_pool_2x2(relu1)														#2*2�ػ������14*14*32��32��14*14���ļ�����

    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)#��2������㣬conv2_w==5*5*1*64
    conv2_b = get_bais([CONV2_KERNEL_NUM]) 							 #CONV1_KERNEL_NUM==64��������ƫִ��������ĺ���,
    conv2 = conv2d(pool1, conv2_w)             					 #pool1�ĳߴ�100*14*14*32��conv2_w==5*5*1*64�������ߴ磬64��14*14�ľ���ÿ��ͼƬ��Ӧ1��14*14*64�ļ�������
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))	 #�������1�ļ������������Լ���
    pool2 = max_pool_2x2(relu2)													 #2*2�ػ������7*7*64��64��14*14���ļ�����

    pool_shape = pool2.get_shape().as_list()       						#�������pool2��ά�ȣ�����list�У�pool_shape)
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]			#nodes==7*7*64==3136��������ĳ���*���*���==������ĸ���
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes]) 			#pool_shape[0]==100����һ��ι���ͼƬ����==batch��������pool2����һ��batch�У�ÿ��nodes����Ķ�ά����
																															#reshaped��100*3136�ľ���ÿ1�д���1��ͼƬ
																															
    fc1_w = get_weight([nodes, FC_SIZE], regularizer)         #ȫ���Ӳ����fc1_w Ϊ3136*512�ľ���
    fc1_b = get_bais([FC_SIZE])																#ƫִ�����512
    fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_w) + fc1_b)				#fc1�ļ�����Ϊ100*512�ľ��󣬴���ÿ��ͼƬ��Ӧ��512����Ԫ����ֵ��100��ͼƬ��

    if train:													#�����ѵ���׶Σ�����һ�ֵ����fc1�������ȥһ�������ļ���������Ԫ��
        fc1 = tf.nn.dropout(fc1, 0.5)

    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)		#ȫ���Ӳ���� fc2_w �� 512*10 �ľ���
    fc2_b = get_bais([OUTPUT_NODE])														#ƫִ�����10
    y = tf.matmul(fc1, fc2_w) + fc2_b                         #fc1==100*512��fc2_w==512*10�� ������,100*10�ľ���100��ͼƬ��
    return y
