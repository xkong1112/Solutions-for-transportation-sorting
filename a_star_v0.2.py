# coding=GBK
import time as tm
from collections import defaultdict

exZero = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5],
          3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8],
          6: [3, 7, 9], 7: [4, 6, 8, 10], 8: [5, 7, 11],
          9: [6, 10], 10: [7, 9, 11], 11: [8, 10]
          }


def mhd(a, b):
    c = divmod(a, 3)  # �ַ�����һλ��0
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


# deepD��Ҫ��¼�ַ��������Ӧ�Ĳ���
# �鿴��һ������λ�������ط������һ��
def judgeEx(a, b):  # a��b�������ַ�����������̸���Ƚ������ַ��������ز���0���Ǹ�Ԫ�ص�ֵ������λ��
    for i in range(len(a)):
        if a[i] != b[i]:
            if a[i] != "0":
                return a[i]
            else:
                return b[i]


def aStar(orgString, aimString):
    # �ֵ��ʼ��
    deepD[orgString] = 0
    preNode[orgString] = -1  # ��¼ǰ���ڵ�
    openT[orgString] = fn(orgString, aimString, deepD[orgString])
    while True:
        curString = min(zip(openT.values(), openT.keys()))[1]
        del (openT[curString])
        if curString == aimString:
            break
        for zero in whereIsZero(curString):
            chaJi = list(set(exZero[zero]).difference(set(whereIsZero(curString))))
            if chaJi != []:  # ������Ϊ��
                for digits in chaJi:  # ���Ϊ�գ������ַ��������ַ������µ��ַ���
                    newString = exc(curString, zero, digits)
                    if preNode.get(newString) == None:
                        deepD[newString] = deepD[curString] + 1  # ���²����ֵ�
                        openT[newString] = fn(newString, aimString, deepD[newString])
                        preNode[newString] = curString
    lst_steps = []
    lst_steps.append(curString)
    while preNode[curString] != -1:  # ����·��
        curString = preNode[curString]
        lst_steps.append(curString)
    lst_steps.reverse()
    return lst_steps


if __name__ == "__main__":
    loopTimes=0
    for i in range (50):
        loopTimes+=1
        openT = {}
        preNode = {}
        deepD = {}
        keysList = []
        listA = []
        orgString = "500431000260"
        aimString = "000123456000"
        start = tm.time()
        printL = aStar(orgString, aimString)
        end = tm.time()
        collision_point = []
        count = 0
        finalSt = defaultdict(list)
        for i in range(len(printL) - 1):  # �����е��ַ�������һ��
            listA.append(judgeEx(printL[i], printL[i + 1]))
        collision_point.append(listA[0])
        for i in range(0, len(listA)):
            if listA[i] not in collision_point:
                finalSt[count].append(printL[i + 1])
                collision_point.append(listA[i])
            else:
                count += 1
                finalSt[count].append(printL[i + 1])
                collision_point.clear()
                collision_point.append(listA[i])
        print("��" + str(loopTimes) + "�Σ�"+"����"+str(count)+"��")


'''
for k, v in finalSt.items():
            print("��" + str(k) + "��")
            for i in range(len(v)):
                print(v[i][:3])
                print(v[i][3:6])
                print(v[i][6:9])
                print(v[i][9:])
                print("��")
'''



'''for i in range(len(c)):
        print("��" + str(i)+"��")
        print(c[i][:3])
        print(c[i][3:6])
        print(c[i][6:9])
        print(c[i][9:])
    print("��ʱ"+str(end-start)+"��")
    keysList=list(openT.keys())
    print(len(keysList))'''
# 000364105020 13 ��
# 500431000260 16 ��
# 040250000361 17 ��
