 
#	  coding:utf-8
#	 @file    backward.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-05 22:08:54
 
 
import tensorflow as tf
import numpy as np
import cifar10_lenet5_forward
import cifar10_lenet5_generateds #1
import os
from tensorflow.examples.tutorials.mnist import input_data


#BATCH_SIZE = 100          		# 一个训练batch中的训练数据个数
BATCH_SIZE = 1          		# 一个训练batch中的训练数据个数
LEARNING_RATE_BASE = 0.005		# 基础学习率
LEARNING_RATE_DECAY = 0.99		# 学习率的衰减率
REGULARIZER = 0.0001 			# 描述模型复杂度的正则化项在损失函数中的系数
STEPS = 50000                           # 训练轮数
MOVING_AVERAGE_DECAY = 0.99		# 滑动平均衰减率
MODEL_SAVE_PATH = "./model"		# 模型存储路径
MODEL_NAME = "cifar10_model"    	# 模型命名

train_num_examples = 50000 		#2训练样本数



def backward():								#执行反向传播，训练参数w
    x = tf.placeholder(tf.float32, [                                    #定义占位符x，以之代替输入图片
        BATCH_SIZE,
        cifar10_lenet5_forward.IMAGE_SIZE,
        cifar10_lenet5_forward.IMAGE_SIZE,
        cifar10_lenet5_forward.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, cifar10_lenet5_forward.OUTPUT_NODE])#定义占位符y_，用来接神经元计算结果
                                                                        #True表示训练阶段，在进行forward时，if语句成立，进行dropout
    y = cifar10_lenet5_forward.forward(x,True, REGULARIZER)		#y是神经元的计算结果，下一步喂给y_
    global_step = tf.Variable(0, trainable = False)                     #定义变量global_step，并把它的属性设置为不可训练  

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,     #用交叉熵ce（cross entropy），使用tf.nn.sparse_softmax_cross_entropy_with_logits函数计算交叉熵，
                                            labels = tf.argmax(y_, 1))  #其第一个参数是神经网络不包括softmax层的前向传播结果，第二个参数是训练数据的正确答案 
                                                                            # tf.argmax(vector, 1)：返回的是vector中的最大值的索引号
    cem = tf.reduce_mean(ce)                                            #计算在当前batch中所有样例的交叉熵平均值 
    loss = cem + tf.add_n(tf.get_collection('losses')) 			# 总损失等于交叉熵损失 + 正则化损失的和,losses保存有正则化的计算结果（forward中getweight（）对参数进行了正则化计算）
    learning_rate = tf.train.exponential_decay(				# 设置指数衰减的学习率
    				               LEARNING_RATE_BASE,  	# 基础学习率，随着迭代的进行，更新变量时使用的学习率在此基础上递减 	
                                               global_step,			    # 当前迭代轮数
                                               train_num_examples / BATCH_SIZE,     # 过完所有训练数据所需迭代次数 	
                                               LEARNING_RATE_DECAY,		    # 指数学习衰减率
                                               staircase = True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)# 使用梯度下降优化损失函数，损失函数包含了交叉熵和正则化损失
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step) 				  # 初始化滑动平均类
    ema_op = ema.apply(tf.trainable_variables()) 																		  	# 对所有表示神经网络参数的变量进行滑动平均
    																												#bind operation train_step & ema_op together to realize two operations at time
    with tf.control_dependencies([train_step, ema_op]):		        # 使用tf.control_dependencies机制一次完成多个操作。在此神经网络模型中，每过一遍数据既通过 
        train_op = tf.no_op(name = 'train')             		# 反向传播更新参数，又更新了每一个参数的滑动平均值

    saver = tf.train.Saver() 																# 声明tf.train.Saver类用于保存模型
        
    img_batch, lable_batch = cifar10_lenet5_generateds.get_tfrecord(BATCH_SIZE, isTrain=True)   #3一次批获取 batch_size 张图片和标签

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())                     # 初始化所有变量
																												# 恢复模块
        ckpt = tf.train.get_checkpoint_state("./model")                 # 从"./model"中加载训练好的模型
        if ckpt and ckpt.model_checkpoint_path: 			# 若ckpt和保存的模型在指定路径中存在，则将保存的神经网络模型加载到当前会话中
            print ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()					#4开启线程协调器
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  #5


        for i in range(STEPS):                                          # 迭代地训练神经网络
            xs, ys = sess.run([img_batch, lable_batch])                 #6将一个batch的训练数数据和对应标签分别赋给xs，ys

            reshaped_xs = np.reshape(xs, (                              #导入部分，更改参数的形状
                BATCH_SIZE,
                cifar10_lenet5_forward.IMAGE_SIZE,
                cifar10_lenet5_forward.IMAGE_SIZE,
                cifar10_lenet5_forward.NUM_CHANNELS))             

            _, loss_value, step = sess.run([train_op, loss,global_step],    # 计算损失函数结果，计算节点train_op, loss,global_step并返回结果至 _, loss_value, step ，
                                            feed_dict = {x : reshaped_xs, y_ : ys})  #'_' means an anonymous variable which will not in use any more
            if i % 1000 == 0:                                               # 每1000轮打印损失函数信息，并保存当前的模型
                print("after %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step = global_step)   # 保存当前模型，globle_step参数可以使每个被保存模型的文件名末尾都加上训练的轮数
      																																															 # 文件的名字是MODEL_SAVE_PATH + MODEL_NAME + global_step
        coord.request_stop()                                                #7关闭线程协调器
        coord.join(threads)                                                 #8
        
def main():
    backward()

if __name__ == '__main__':                                                   #main function,
    main()

