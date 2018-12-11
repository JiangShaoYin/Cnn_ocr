#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
身份证文字+数字生成类

@author: pengyuanjie
"""
import numpy as np
import freetype
import copy
import random
import cv2

class put_chinese_text(object):
    def __init__(self, ttf):
        self._face = freetype.Face(ttf)

    def draw_text(self, image, pos, text, text_size, text_color):
        self._face.set_char_size(text_size * 64)
        metrics = self._face.size
        ascender = metrics.ascender/64.0

        ypos = int(ascender)

        if not isinstance(text, unicode):
            text = text.decode('utf-8')
        img = self.draw_string(image, pos[0], pos[1]+ypos, text, text_color)
        return img

    def draw_string(self, img, x_pos, y_pos, text, color):
        prev_char = 0
        pen = freetype.Vector()
        pen.x = x_pos << 6   # div 64
        pen.y = y_pos << 6

        hscale = 1.0
        matrix = freetype.Matrix(int(hscale)*0x10000L, int(0.2*0x10000L),\
                                 int(0.0*0x10000L), int(1.1*0x10000L))
        cur_pen = freetype.Vector()
        pen_translate = freetype.Vector()

        image = copy.deepcopy(img)
        for cur_char in text:
            self._face.set_transform(matrix, pen_translate)

            self._face.load_char(cur_char)
            kerning = self._face.get_kerning(prev_char, cur_char)
            pen.x += kerning.x
            slot = self._face.glyph
            bitmap = slot.bitmap

            cur_pen.x = pen.x
            cur_pen.y = pen.y - slot.bitmap_top * 64
            self.draw_ft_bitmap(image, bitmap, cur_pen, color)

            pen.x += slot.advance.x
            prev_char = cur_char
#        print "draw_string():",image
        return image

    def draw_ft_bitmap(self, img, bitmap, pen, color):
        x_pos = pen.x >> 6
        y_pos = pen.y >> 6
        cols = bitmap.width
        rows = bitmap.rows

        glyph_pixels = bitmap.buffer

        for row in range(rows):
            for col in range(cols):
                if glyph_pixels[row*cols + col] != 0:
                    img[y_pos + row][x_pos + col][0] = color[0]
                    img[y_pos + row][x_pos + col][1] = color[1]
                    img[y_pos + row][x_pos + col][2] = color[2]

#生成ID的类
class gen_id_card(object):      #生成
    def __init__(self):         #初始化类的成员
       #self.words = open('AllWords.txt', 'r').read().split(' ')
       self.number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
       self.char_set = self.number  #char_set是个list,内含0~9十个元素
       self.len = len(self.char_set)#长度为10
       
       self.max_size = 18
       self.ft = put_chinese_text('fonts/OCR-B.ttf') #类的成员ft为1个对象，该对象根据指定字体，生成图片
       
    #随机生成字串，长度固定,返回text,及对应的向量
    def random_text(self):
        text = ''
        vecs = np.zeros((self.max_size * self.len))     #1*180的数组
        #size = random.randint(1, self.max_size)
        size = self.max_size                            #size == 18
        for i in range(size):
            c = random.choice(self.char_set)            #随机成成一个0-9的数，例如  c = 4
            #print "c:",c
            vec = self.char2vec(c)                      #将4编程一个向量
            #print "vec:",vec
            text = text + c                             #text=''，text是一个string,所以4+2=42
            #print 'text:',text
            vecs[i*self.len:(i+1)*self.len] = np.copy(vec)
        return text,vecs                                # vecs是一个1*180的数组
    
    #根据生成的text，生成image,返回标签string和图片向量vecs
    def gen_image(self):
        text,vecs = self.random_text()
        img = np.zeros([32,256,3])
        color_ = (255,255,255) # pure white
        pos = (0, 0)
        text_size = 21
        image = self.ft.draw_text(img, pos, text, text_size, color_)    #iamge尺寸(32, 256, 3)
       # print image.shape,image[:,:,2].shape
       # 仅返回单通道值，颜色对于汉字识别没有什么意义
        return image[:,:,2],text,vecs                                   #image[:,:,2]的尺寸(32, 256)

    #字转向量
    def char2vec(self, c):
        vec = np.zeros((self.len))       #长度为10的全0的数组
        for j in range(self.len):
            if self.char_set[j] == c:    #判断c在[0,1,2......,9]的这个list里面的下标,以c=4为例char_set[3]=4,下一步的vec[3]=1     
                vec[j] = 1
        return vec                       #vec= [0,0,0,1,.....0,0]
        
    #向量转文本,[180]转长度为18的字符串178221255607512060
    def vec2text(self, vecs):
        text = ''
        v_len = len(vecs)
        for i in range(v_len):
            if(vecs[i] == 1):
                text = text + self.char_set[i % self.len]
        print 'vec2text:text=',text
        return text                     #text是一个string，等于178221255607512060 

if __name__ == '__main__':
    for i in range(5000):
        ID = gen_id_card()
        image,text,vec = ID.gen_image()
        
        name = "./pic/" + str(i) + " " + text + ".jpg" 
        cv2.imwrite(name, image)
#    cv2.imshow('image', image)
#    cv2.waitKey(0)                     #一直显示图像，直到键入0为之
