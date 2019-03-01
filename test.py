 
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
    with tf.Graph().as_default() as g:                                      #����֮ǰ����ļ���ͼ����ִ�����²���
        x = tf.placeholder(tf.float32, [                                    #����ռλ��x����֮��������ͼƬ
            BATCH_SIZE,
            forward.IMAGE_HEIGHT,
            forward.IMAGE_WIDTH,
            forward.NUM_CHANNELS])
        
        y_ = tf.placeholder(tf.float32,[None, 10])                          #����ռλ��y_�����������ݼ��еı�ǩֵ
        y = forward.forward(x, False,  None)                                #y����Ԫ�ļ�����
        y = tf.reshape(y, [-1, 10])
        predict_ans = tf.argmax(y,1)                                               #batch*18������

        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)# ʵ�ֻ���ƽ��ģ�ͣ�����MOVING_AVERAGE_DECAY���ڿ���ģ�͸��µ��ٶȣ�ѵ�������л��ÿһ������ά��һ��Ӱ�ӱ���
        ema_restore = ema.variables_to_restore()                                      # variable_to_restore()����dict ({ema_variables : variables})���ֵ��б��������Ӱ��ֵ����ֵ
        saver = tf.train.Saver(ema_restore) 			                                    # �����ɻ�ԭ����ƽ��ֵ�Ķ���saver������ʱʹ��w��Ӱ��ֵ���и��õ�������
         
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))              # �Ƚ�Ԥ��ֵ�ͱ�׼����õ�correct_prediction��if tf.argmax(y, 1) equals to tf.argmax(y_, 1),correct_prediction will be set True
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))            # ��correct_prediction��ֵ��boolean��תΪtf.float32�ͣ����ֵ���ó�Ԥ��׼ȷ�� 

        img_batch,label_batch = generateds.get_tfrecord(TEST_NUM, isTrain=True)  #2 һ������ȡ TEST_NUM ��ͼƬ�ͱ�ǩ
        

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)    # ��ָ��·���У�����ѵ���õ�ģ��
                if ckpt and ckpt.model_checkpoint_path:                                   # ������ckptģ����ִ�����»ָ�����
                    saver.restore(sess, ckpt.model_checkpoint_path)                       # �ָ��Ự����ǰ��������
     
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] # ��ckpt.model_checkpoint_path�У�ͨ���ַ� "/" �� "-"��ȡ�����һ����������������������ָ�������

                    coord = tf.train.Coordinator()                                        #3�����߳�Э����
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)        #4
                    xs,ys = sess.run([img_batch, label_batch])                            #5# �� sess.run ��ִ��ͼƬ�ͱ�ǩ������ȡ

                    reshaped_xs = np.reshape(xs, (                              #���벿�֣����Ĳ�������״
                        BATCH_SIZE,
                        forward.IMAGE_HEIGHT,
                        forward.IMAGE_WIDTH,
                        forward.NUM_CHANNELS))             

                    reshaped_ys = np.reshape(ys, (-1,10))
#                    print y_,reshaped_ys
#                    print x, reshaped_xs
                    accuracy_score,predict_value = sess.run([accuracy,predict_ans], # ����׼ȷ��
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
            time.sleep(INTERVAL_TIME)                                             # ���õȴ�ʱ�䣬��backward�����µ�checkpoint�ļ�����ѭ��ִ��test����

def main():
    test()
    
#main function,
if __name__ == '__main__':
    main()
