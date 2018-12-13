 
#        coding:utf-8
#	 @file    filename.py
#	 @author  1801210547_江绍印(jiangshaoyin@pku.edu.cn)
#	 @date    2018-12-13 09:30:57
 
 
import tensorflow as tf
import numpy as np
import os

files = os.listdir("./pic/train")
for filename in files:
    text =  filename.split()

print files
