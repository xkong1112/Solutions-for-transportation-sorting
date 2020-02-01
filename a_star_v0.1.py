#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:12:12 2019

@author: yu
"""

# coding=GBK
import time as tm
from collections import defaultdict

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


def mhd(a, b):
    c = divmod(a, 3)  # 字符串第一位是0
    d = divmod(b, 3)
    g = abs(c[0] - d[0]) + abs(c[1] - d[1])
    return g


def exc(a, i, j):
    if i > j:
        i, j = j, i
    b = a[:i] + a[j] + a[i + 1:j] + a[i] + a[j + 1:]
    return b


def whereIsZero(curString):
    a = []
    for i in range(len(curString)):
        if curString[i] == "0":
            a.append(i)
    return a


def fn(curString, aimString, d):
    e = 0
    for i in range(0, len(curString)):
        if curString[i] != "0":
            if curString[i] != aimString[i]:
                b = mhd(i, aimString.index(curString[i]))
                e += b

    f = e + d
    return f


# deepD需要记录字符串和其对应的步数
# 查看不一样的两位数并返回非零的那一个
def judgeEx(a, b):  # a和b是两个字符串，这个方程负责比较两个字符串并返回不是0的那个元素的值（不是位置
    for i in range(len(a)):
        if a[i] != b[i]:
            if a[i] != "0":
                return a[i]
            else:
                return b[i]


def aStar(orgString, aimString):
    # 字典初始化
    deepD[orgString] = 0
    preNode[orgString] = -1  # 记录前驱节点
    openT[orgString] = fn(orgString, aimString, deepD[orgString])
    while True:
        curString = min(zip(openT.values(), openT.keys()))[1]
        del (openT[curString])
        if curString == aimString:
            break
        for zero in whereIsZero(curString):
            chaJi = list(set(exZero[zero]).difference(set(whereIsZero(curString))))
            if chaJi != []:  # 如果差集不为空
                for digits in chaJi:  # 差集不为空，两个字符串交换字符生成新的字符串
                    newString = exc(curString, zero, digits)
                    if preNode.get(newString) == None:
                        deepD[newString] = deepD[curString] + 1  # 更新步数字典
                        openT[newString] = fn(newString, aimString, deepD[newString])
                        preNode[newString] = curString
    lst_steps = []
    lst_steps.append(curString)
    while preNode[curString] != -1:  # 存入路径
        curString = preNode[curString]
        lst_steps.append(curString)
    lst_steps.reverse()
    return lst_steps


if __name__ == "__main__":
    orgString = "000364105020"
    aimString = "000123456000"
    start = tm.time()
    printL = aStar(orgString, aimString)
    end = tm.time()
    diPoint = []
    count = 0
    finalSt = defaultdict(list)
    for i in range(len(printL) - 1):  # 把所有的字符串遍历一下
        listA.append(judgeEx(printL[i], printL[i + 1]))
    print(listA)
    diPoint.append(listA[0])
    for i in range(0, len(listA)):
        if listA[i] not in diPoint:
            finalSt[count].append(printL[i + 1])
            diPoint.append(listA[i])
        else:
            count += 1
            finalSt[count].append(printL[i + 1])
            diPoint.clear()
            diPoint.append(listA[i])

    for k, v in finalSt.items():
        print("第" + str(k) + "步")
        for i in range(len(v)):
            print(v[i][:3])
            print(v[i][3:6])
            print(v[i][6:9])
            print(v[i][9:])
            print("→")

    '''for i in range(len(c)):
        print("第" + str(i)+"步")
        print(c[i][:3])
        print(c[i][3:6])
        print(c[i][6:9])
        print(c[i][9:])
    print("用时"+str(end-start)+"秒")
    keysList=list(openT.keys())
    print(len(keysList))'''
# 000364105020 13 √
# 500431000260 16 √
# 040250000361 17 √
