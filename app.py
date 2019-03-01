 
#	  coding:utf-8
#	 @file    app.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-07 16:16:16
 
 
import tensorflow as tf
import numpy as np
import forward
import backward
import generateds
import os
from PIL import Image
testNum = 10


def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((256,32),Image.ANTIALIAS)    #��������ݵķ�����ԭͼ����Ϊ32*32��antialias������ݣ�
    im_array = np.array(reIm.convert('L'))        #��ͼƬתΪ�Ҷ�ͼ��0�����ڣ�255������
#    print im_array.shape
    threshold = 50                                #>��ֵ������Ϊ255,<��ֵ������Ϊ0
    for i in range(32):                           #width 32
        for j in range(256):                       #height 256
       #     im_array[i][j] = 255 - im_array[i][j] #������ͼȡ�������ֺ�backwardѵ��ͼƬ��ʽ��һ�£�
            if (im_array[i][j] < threshold):
                im_array[i][j] = 0
            else:      
                im_array[i][j] = 255 
    reshaped_array = im_array.reshape([1,32,256,1])
    reshaped_array = reshaped_array.astype(np.float32) 		#����������Ϊ������
    img_ready = np.multiply(reshaped_array, 1.0/255)			#��0~255��intֵ������Ϊ0~1�ĸ���ֵ
    return img_ready


def restore_model(testPicArr):                     # �����ؽ�ģ�Ͳ�����Ԥ��
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [1,32,256,1])	#����ռλ��x����֮��������ͼƬ
        y = forward.forward(x, False,  None)                                #y����Ԫ�ļ�����
        y = tf.reshape(y, [-1, 10])
        predict_Value = tf.argmax(y,1)                #����1��18�е�1ά���飬                               #batch*18������

        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY) # ʵ�ֻ���ƽ��ģ�ͣ�����MOVING_AVERAGE_DECAY���ڿ���ģ�͸��µ�
                                                                                       # �ٶȣ�ѵ�������л��ÿһ������ά��һ��Ӱ�ӱ���
        ema_restore = ema.variables_to_restore()          # variable_to_restore()����dict ({ema_variables : variables})���ֵ��б��������Ӱ��ֵ����ֵ
        saver = tf.train.Saver(ema_restore)               # �����ɻ�ԭ����ƽ��ֵ�Ķ���saver������ʱʹ��w��Ӱ��ֵ���и��õ�������

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH) # ��ָ��·���У�����ѵ���õ�ģ��
            if ckpt and ckpt.model_checkpoint_path:                                # ������ckptģ����ִ�����»ָ�����
                saver.restore(sess, ckpt.model_checkpoint_path)                    # �ָ��Ự����ǰ��������
                predictValue = sess.run(predict_Value, feed_dict={x:testPicArr})    #ͨ��feed_dict������ͼƬ���룬���Ԥ����
                print "predictValue",predictValue.shape
                return predictValue
            else:                                                                  # û�гɹ�����ckptģ�ͣ�
                print "no checkpoint file found"
                return -1

def app():
    test_list = os.listdir("./test")
    for testPic in test_list:													#testNum==10
        testPic = os.path.join("./test",testPic)				#�ַ���ƴ�ӣ��������ļ���·��
        testPicArr = pre_pic(testPic)									#��ͼƬ��Ԥ����
        predictValue = restore_model(testPicArr)			#�������ͼƬԤ��ֵ
        print "the predict number is :",predictValue

def main():
    app()

#execute main funcion
if __name__ == '__main__':
    main()
