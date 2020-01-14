# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 18:37:31 2020

@author: 98212
"""
import numpy as np
class Maze():
    def __init__(self,curString, aimString):
        self.curString =curString
        self.aimString = aimString
    def add(self):
        self.curString = 1+self.aimString
        self.aimString += self.curString 
        return self.curString,self.aimString
        #print(self.curString,self.aimString)
    def add2(self):
        print(self.add()[0])
a=Maze(111,222)

c='0000123456000'
b=np.fromstring(c, dtype=np.uint8) # to string 用于 把数组转换成为字符串， from string把字符串转换成为数组
#print(b)                             # 报错： string size must be a multiple of element size，于是声明dtype=np.uint8,编程课uint8的ASCII码
#print(b.tostring())

e=c.split() # e变成了list 输出['0123456']
f=np.array(c.split()) # e变成了list 输出['0123456']
g=np.array(''.join(c.split())) #输出0123456
print(g)
h=np.array([0,0,0,1,2,3,4,5,6,0,0,0])#I think when you start a literal number with a 0, it interprets it as an octal number and you can't have an '8' in an octal number.
i=np.array(','.join(c.split())) #输出0123456
print(h)
print(i)