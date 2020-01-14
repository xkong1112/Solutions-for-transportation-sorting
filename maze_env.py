#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:12:52 2019
@author: yu

Reinforcement learning approach for solving sorting problem.
Manhattan Distance -1   [reward = -1].
Manhattan Distance +1   [reward = +1].
"""
class Maze():#给环境一个输入的字符串（state），可以选择动作，返回各个动作的奖励值hn并且给出下一个新的state
    openT = {}
    preNode = {}
    deepD = {}
    keysList = []
    listA = []
    exZero = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5],
              3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8],
              6: [3, 7, 9], 7: [4, 6, 8, 10], 8: [5, 7, 11],
              9: [6, 10], 10: [7, 9, 11], 11: [8, 10]
              }

    def __init__(self,curString, aimString):
        self.curString =curString
        self.aimString = aimString

    def mhd(self, a, b):
        c = divmod(a, 3)  # 字符串第一位是0
        d = divmod(b, 3)
        g = abs(c[0] - d[0]) + abs(c[1] - d[1])
        return g

    def exc(self, i, j):# 输入的 i，j 是两个数字，或者说序号
        if i > j:
            i, j = j, i
        b = self.curString[:i] + self.curString[j] + self.curString[i + 1:j] + self.curString[i] + self.curString[j + 1:]
        return b

    def whereIsZero(self): # 输出的是0所在的位置对应的序号
        a = []
        for i in range(len(self.curString)):
            if self.curString[i] == "0":
                a.append(i)
        return a

    def hn_(self):
        for i in range(0, len(self.curString)):
            if self.curString[i] != "0":
                if self.curString[i] != self.aimString[i]:
                    b = self.mhd(i, self.aimString.index(self.curString[i]))
        return b


    def step(self): #相当于A star的一步不包括更新。
        # Actions
        # 先给变量赋值，然后再调用，相当于对临时变量进行运维
        # reward就是曼哈顿距离减少的多少,但是观测的是当前的state，那reward对应的是啥
        # 每走一步都是-1然后hn也mhd距离越大也是越差，整个给他最小化
        # fxreward就是反向reward的意思，主要的原因是tf只能minimize，那就把这个反向奖励minimize就完事了
        obs=self.curString # s_
        fxreward = self.hn_(self.curString, self.aimString)+self.action()[1]
        if obs == self.aimString: #or self.action()[1] > 10000: # count
            done  = True
        else:
            done = False
            return obs,fxreward,done #s_, hn , done
        
    def action(self): # 所有可能的action以及其对应的reward的集合
        chaJi = []
        count = None
        #s_={}
        s_=[]
        for zero in self.whereIsZero(self.curString):
            chaJi = list(set(self.exZero[zero]).difference(set(self.whereIsZero(self.curString))))
            if chaJi != []:  # 如果差集不为空
                for digits in chaJi:  # 差集不为空，两个字符串交换字符生成新的字符串
                    newString = self.exc(self.curString, zero, digits) # 随机交换一个非零数字和零
                    #delta_hn = self.hn_(self.curString, self.aimString)-self.hn_(newString, self.aimString)
                    #s_[newString] = delta_hn
                    s_=s_.append(newString)
                    count += 1
                    return s_, count #, done
        
    
'''
#定义环境的更新
def update():
    for t in range(100):
        s = env.reset()
        while True:
            # env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break
if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
    '''