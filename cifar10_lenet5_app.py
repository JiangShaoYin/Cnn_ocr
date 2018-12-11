 
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
    reIm = img.resize((32,32),Image.ANTIALIAS)    #用消除锯齿的方法将原图调整为32*32（antialias：抗锯齿）
    im_array = np.array(reIm.convert('RGB'))        #将图片转为灰度图，0代表纯黑，255代表纯白
    '''threshold = 50                                #>阈值，设置为255,<阈值，设置为0
    for i in range(32):                           #width 32
        for j in range(32):                       #height 32
            im_array[i][j] = 255 - im_array[i][j] #将测试图取反（保持和backward训练图片格式的一致）
            if (im_array[i][j] < threshold):
                im_array[i][j] = 0
            else:      
                im_array[i][j] = 255 '''
    reshaped_array = np.reshape(im_array, (                              #导入部分，更改参数的形状
                        1,                                              #测试时，每次输入一张图片
                        cifar10_lenet5_forward.IMAGE_SIZE,
                        cifar10_lenet5_forward.IMAGE_SIZE,
                        cifar10_lenet5_forward.NUM_CHANNELS))             
    reshaped_array = reshaped_array.astype(np.float32) 		#将整数调整为浮点数
    img_ready = np.multiply(reshaped_array, 1.0/255)			#将0~255的int值，调整为0~1的浮点值
    return img_ready


def restore_model(testPicArr):                                              # 定义重建模型并进行预测
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [1,32,32,3])	#定义占位符x，以之代替输入图片
        y = cifar10_lenet5_forward.forward(x,False, None)																# 对x执行前向传播得到输出y
        predictValue = tf.argmax(y,1)                                       # 定义输入图片的预测值

        ema = tf.train.ExponentialMovingAverage(cifar10_lenet5_backward.MOVING_AVERAGE_DECAY) # 实现滑动平均模型，参数MOVING_AVERAGE_DECAY用于控制模型更新的
                                                                                       # 速度，训练过程中会对每一个变量维护一个影子变量
        ema_restore = ema.variables_to_restore()          # variable_to_restore()返回dict ({ema_variables : variables})，字典中保存变量的影子值和现值
        saver = tf.train.Saver(ema_restore)               # 创建可还原滑动平均值的对象saver，测试时使用w的影子值，有更好的适配性

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(cifar10_lenet5_backward.MODEL_SAVE_PATH) # 从指定路径中，加载训练好的模型
            if ckpt and ckpt.model_checkpoint_path:                                # 若已有ckpt模型则执行以下恢复操作
                saver.restore(sess, ckpt.model_checkpoint_path)                    # 恢复会话到当前的神经网络
                predictValue = sess.run(predictValue, feed_dict={x:testPicArr})    #通过feed_dict将测试图片输入，输出预测结果
                return predictValue
            else:                                                                  # 没有成功加载ckpt模型，
                print "no checkpoint file found"
                return -1

def app():
    for i in range(testNum):													#testNum==10
        testPic = "./pic/" + str(i) + ".jpg"					#字符串拼接，作测试文件的路径
        testPicArr = pre_pic(testPic)									#做图片的预处理
        predictValue = restore_model(testPicArr)			#计算测试图片预测值
        print "the predict number is :%d"%predictValue

def main():
    app()

#execute main funcion
if __name__ == '__main__':
    main()
