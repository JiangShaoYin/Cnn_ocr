 
#        coding:utf-8
#	 @file    filename.py
#	 @author  1801210547_江绍印(jiangshaoyin@pku.edu.cn)
#	 @date    2018-12-13 09:30:57
 
 
import tensorflow as tf
import numpy as np
import os

def Text2Vecs(text):#将18位的字符串387230452075106773转化成180列的向量
    chars = list(text)#chars = ['3', '8', '7', '2', '3', '0', '4', '5', '2', '0', '7', '5', '1', '0', '6', '7', '7', '3']
    base =[0]*10
    vecs = []
    i = 1
    for char in chars:
        base =[0]*10
        base[int(char)] = 1 #base[3]= 1
        vecs += base
    return vecs

def vec2text(vecs):
    text = ''
    char_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    v_len = len(vecs)        #vecs为180列的数组
    for i in range(v_len):
        if(vecs[i] == 1):   
            text = text + char_set[i % 10]
    return text      #text是一个string，等于178221255607512060 

files = os.listdir("./pic/train")
for filename in files:
    text1 =  filename.split()[1]
    vecs = Text2Vecs(text1)
    text2 = vec2text(vecs)
    print text1,vecs,text2
    break



