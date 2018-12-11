 
#	  coding:utf-8
#	 @file    app.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-07 16:16:16
 
 
import tensorflow as tf
import numpy as np
import cifar10_lenet5_forward
import cifar10_lenet5_backward
from PIL import Image
testNum = 10


def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((32,32),Image.ANTIALIAS)    #��������ݵķ�����ԭͼ����Ϊ32*32��antialias������ݣ�
    im_array = np.array(reIm.convert('RGB'))        #��ͼƬתΪ�Ҷ�ͼ��0�����ڣ�255������
    '''threshold = 50                                #>��ֵ������Ϊ255,<��ֵ������Ϊ0
    for i in range(32):                           #width 32
        for j in range(32):                       #height 32
            im_array[i][j] = 255 - im_array[i][j] #������ͼȡ�������ֺ�backwardѵ��ͼƬ��ʽ��һ�£�
            if (im_array[i][j] < threshold):
                im_array[i][j] = 0
            else:      
                im_array[i][j] = 255 '''
    reshaped_array = np.reshape(im_array, (                              #���벿�֣����Ĳ�������״
                        1,                                              #����ʱ��ÿ������һ��ͼƬ
                        cifar10_lenet5_forward.IMAGE_SIZE,
                        cifar10_lenet5_forward.IMAGE_SIZE,
                        cifar10_lenet5_forward.NUM_CHANNELS))             
    reshaped_array = reshaped_array.astype(np.float32) 		#����������Ϊ������
    img_ready = np.multiply(reshaped_array, 1.0/255)			#��0~255��intֵ������Ϊ0~1�ĸ���ֵ
    return img_ready


def restore_model(testPicArr):                                              # �����ؽ�ģ�Ͳ�����Ԥ��
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [1,32,32,3])	#����ռλ��x����֮��������ͼƬ
        y = cifar10_lenet5_forward.forward(x,False, None)																# ��xִ��ǰ�򴫲��õ����y
        predictValue = tf.argmax(y,1)                                       # ��������ͼƬ��Ԥ��ֵ

        ema = tf.train.ExponentialMovingAverage(cifar10_lenet5_backward.MOVING_AVERAGE_DECAY) # ʵ�ֻ���ƽ��ģ�ͣ�����MOVING_AVERAGE_DECAY���ڿ���ģ�͸��µ�
                                                                                       # �ٶȣ�ѵ�������л��ÿһ������ά��һ��Ӱ�ӱ���
        ema_restore = ema.variables_to_restore()          # variable_to_restore()����dict ({ema_variables : variables})���ֵ��б��������Ӱ��ֵ����ֵ
        saver = tf.train.Saver(ema_restore)               # �����ɻ�ԭ����ƽ��ֵ�Ķ���saver������ʱʹ��w��Ӱ��ֵ���и��õ�������

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(cifar10_lenet5_backward.MODEL_SAVE_PATH) # ��ָ��·���У�����ѵ���õ�ģ��
            if ckpt and ckpt.model_checkpoint_path:                                # ������ckptģ����ִ�����»ָ�����
                saver.restore(sess, ckpt.model_checkpoint_path)                    # �ָ��Ự����ǰ��������
                predictValue = sess.run(predictValue, feed_dict={x:testPicArr})    #ͨ��feed_dict������ͼƬ���룬���Ԥ����
                return predictValue
            else:                                                                  # û�гɹ�����ckptģ�ͣ�
                print "no checkpoint file found"
                return -1

def app():
    for i in range(testNum):													#testNum==10
        testPic = "./pic/" + str(i) + ".jpg"					#�ַ���ƴ�ӣ��������ļ���·��
        testPicArr = pre_pic(testPic)									#��ͼƬ��Ԥ����
        predictValue = restore_model(testPicArr)			#�������ͼƬԤ��ֵ
        print "the predict number is :%d"%predictValue

def main():
    app()

#execute main funcion
if __name__ == '__main__':
    main()
