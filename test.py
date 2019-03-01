 
#	 coding:utf-8
#	 @file    cifar10_lenet5test.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-06 11:18:00
 
 
import tensorflow as tf
import numpy as np
import time
import forward
import backward
import generateds
INTERVAL_TIME = 5
TEST_NUM = 10 #1
BATCH_SIZE = 10
#test!

def test():
    with tf.Graph().as_default() as g:                                      #复现之前定义的计算图，并执行以下操作
        x = tf.placeholder(tf.float32, [                                    #定义占位符x，以之代替输入图片
            BATCH_SIZE,
            forward.IMAGE_HEIGHT,
            forward.IMAGE_WIDTH,
            forward.NUM_CHANNELS])
        
        y_ = tf.placeholder(tf.float32,[None, 10])                          #定义占位符y_，用来接数据集中的标签值
        y = forward.forward(x, False,  None)                                #y是神经元的计算结果
        y = tf.reshape(y, [-1, 10])
        predict_ans = tf.argmax(y,1)                                               #batch*18行数据

        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)# 实现滑动平均模型，参数MOVING_AVERAGE_DECAY用于控制模型更新的速度，训练过程中会对每一个变量维护一个影子变量
        ema_restore = ema.variables_to_restore()                                      # variable_to_restore()返回dict ({ema_variables : variables})，字典中保存变量的影子值和现值
        saver = tf.train.Saver(ema_restore) 			                                    # 创建可还原滑动平均值的对象saver，测试时使用w的影子值，有更好的适配性
         
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))              # 比较预测值和标准输出得到correct_prediction，if tf.argmax(y, 1) equals to tf.argmax(y_, 1),correct_prediction will be set True
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))            # 将correct_prediction的值从boolean型转为tf.float32型，求均值，得出预测准确率 

        img_batch,label_batch = generateds.get_tfrecord(TEST_NUM, isTrain=True)  #2 一次批获取 TEST_NUM 张图片和标签
        

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)    # 从指定路径中，加载训练好的模型
                if ckpt and ckpt.model_checkpoint_path:                                   # 若已有ckpt模型则执行以下恢复操作
                    saver.restore(sess, ckpt.model_checkpoint_path)                       # 恢复会话到当前的神经网络
     
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] # 从ckpt.model_checkpoint_path中，通过字符 "/" 和 "-"提取出最后一个整数（保存的轮数），恢复轮数，

                    coord = tf.train.Coordinator()                                        #3开启线程协调器
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)        #4
                    xs,ys = sess.run([img_batch, label_batch])                            #5# 在 sess.run 中执行图片和标签的批获取

                    reshaped_xs = np.reshape(xs, (                              #导入部分，更改参数的形状
                        BATCH_SIZE,
                        forward.IMAGE_HEIGHT,
                        forward.IMAGE_WIDTH,
                        forward.NUM_CHANNELS))             

                    reshaped_ys = np.reshape(ys, (-1,10))
#                    print y_,reshaped_ys
#                    print x, reshaped_xs
                    accuracy_score,predict_value = sess.run([accuracy,predict_ans], # 计算准确率
                                                     feed_dict={x:reshaped_xs,
                                                                y_:reshaped_ys})
#                    print "predict_value:",predict_value
                    print ("after %s training step(s), test accuracy = %g"
                            % (global_step, accuracy_score))

                    coord.request_stop()                                          #6
                    coord.join(threads)                                           #7

                else:                                                             #can not get checkpoint file ,print error infomation
                    print ("No checkpoint file found")
                    return
            time.sleep(INTERVAL_TIME)                                             # 设置等待时间，等backward生成新的checkpoint文件，再循环执行test函数

def main():
    test()
    
#main function,
if __name__ == '__main__':
    main()
