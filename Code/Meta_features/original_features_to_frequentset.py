# -*- coding: utf-8 -*-
import os
from lxml import etree
from threading import *
from time import sleep
import json
import xlrd
from xlrd import xldate_as_tuple
import datetime
import numpy as np
import pandas as pd
import time
import csv
import io
from scapy.utils import PcapReader as PR
from scapy.all import rdpcap
from scapy.all import wrpcap
from scapy.all import PcapWriter
import sys
import scapy;
# from scapy_ssl_tls.ssl_tls import TLS;
import random
# import dataframe_image as dfi
# from PIL import Image
import cv2
import gc


# FP树类
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue  # 节点元素名称，在构造时初始化为给定值
        self.count = numOccur  # 出现次数，在构造时初始化为给定值
        self.nodeLink = None  # 指向下一个相似节点的指针，默认为None
        self.parent = parentNode  # 指向父节点的指针，在构造时初始化为给定值
        self.children = {}  # 指向子节点的字典，以子节点的元素名称为键，指向子节点的指针为值，初始化为空字典

        # 增加节点的出现次数值

    def inc(self, numOccur):
        self.count += numOccur

        # 输出节点和子节点的FP树结构

    def disp(self, ind=1):
        print("-" * ind, self.name, " ", self.count)
        for child in self.children.values():
            child.disp(ind + 1)


# =======================================================构建FP树==================================================


# 对不是第一个出现的节点，更新头指针块。就是添加到相似元素链表的尾部
def updateHeader(nodeToTest, targetNode):
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


# 根据一个排序过滤后的频繁项更新FP树
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        # 有该元素项时计数值+1
        inTree.children[items[0]].inc(count)
    else:
        # 没有这个元素项时创建一个新节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 更新头指针表或前一个相似元素项节点的指针指向新节点
        if headerTable[items[0]][1] == None:
            # 如果是第一次出现，则在头指针表中增加对该节点的指向
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])

    if len(items) > 1:
        # 对剩下的元素项迭代调用updateTree函数
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


# 主程序。创建FP树。dataSet为事务集，为一个字典，键为每个事物，值为该事物出现的次数。minSup为最低支持度
def createTree(dataSet, minSup=1):  # 第一次遍历数据集，创建头指针表
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 移除不满足最小支持度的元素项
    keys = list(headerTable.keys())  # 因为字典要求在迭代中不能修改，所以转化为列表
    for k in keys:
        if headerTable[k] < minSup:
            del headerTable[k]
    # 空元素集，返回空
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None  # 增加一个数据项，用于存放指向相似元素项指针
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]  # 每个键的值，第一个为个数，第二个为下一个节点的位置
    retTree = treeNode("Null Set", 1, None)  # 根节点# 第二次遍历数据集，创建FP树
    for tranSet, count in dataSet.items():
        localD = {}  # 记录频繁1项集的全局频率，用于排序
        for item in tranSet:
            if item in freqItemSet:  # 只考虑频繁项
                localD[item] = headerTable[item][0]  # 注意这个[0]，因为之前加过一个数据项
        if len(localD) > 0:
            orderedItems = [
                v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)
            ]  # 排序
            updateTree(orderedItems, retTree, headerTable, count)  # 更新FP树
    return retTree, headerTable


# =================================================查找元素条件模式基===============================================

# 直接修改prefixPath的值，将当前节点leafNode添加到prefixPath的末尾，然后递归添加其父节点。
# prefixPath就是一条从treeNode（包括treeNode）到根节点（不包括根节点）的路径
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


# 为给定元素项生成一个条件模式基（前缀路径）。basePet表示输入的频繁项，treeNode为当前FP树中对应的第一个节点
# 函数返回值即为条件模式基condPats，用一个字典表示，键为前缀路径，值为计数值。
def findPrefixPath(basePat, treeNode):
    condPats = {}  # 存储条件模式基
    while treeNode != None:
        prefixPath = []  # 用于存储前缀路径
        ascendTree(treeNode, prefixPath)  # 生成前缀路径
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count  # 出现的数量就是当前叶子节点的数量
        treeNode = treeNode.nodeLink  # 遍历下一个相同元素
    return condPats


# =================================================递归查找频繁项集===============================================
# 根据事务集获取FP树和频繁项。
# 遍历频繁项，生成每个频繁项的条件FP树和条件FP树的频繁项
# 这样每个频繁项与他条件FP树的频繁项都构成了频繁项集

# inTree和headerTable是由createTree()函数生成的事务集的FP树。
# minSup表示最小支持度。
# preFix请传入一个空集合（set([])），将在函数中用于保存当前前缀。
# freqItemList请传入一个空列表（[]），将用来储存生成的频繁项集。
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):  # 对频繁项按出现的数量进行排序进行排序
    sorted_headerTable = sorted(
        headerTable.items(), key=lambda p: p[1][0], reverse=True
    )  # 返回重新排序的列表。每个元素是一个元组，[（key,[num,treeNode],()）
    bigL = [v[0] for v in sorted_headerTable]  # 获取频繁项
    for basePat in bigL:
        newFreqSet = preFix.copy()  # 新的频繁项集
        newFreqSet.add((basePat, headerTable[basePat][0]))  # 当前前缀添加一个新元素

        freqItemList.append(newFreqSet)  # 所有的频繁项集列表
        condPattBases = findPrefixPath(
            basePat, headerTable[basePat][1]
        )  # 获取条件模式基。就是basePat元素的所有前缀路径。它像一个新的事务集
        # print(basePat, condPattBases)
        myCondTree, myHead = createTree(condPattBases, minSup)  # 创建条件FP树
        if myHead != None:
            # 用于测试
            # print("conditional tree for:", newFreqSet)
            # myCondTree.disp()
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)  # 递归直到不再有元素


# 生成数据集
def loadSimpDat():
    """
    simpDat = [
        ["r", "z", "h", "j", "p"],
        ["z", "y", "x", "w", "v", "u", "t", "s"],
        ["z"],
        ["r", "x", "n", "o", "s"],
        ["y", "r", "x", "z", "q", "t", "p"],
        ["y", "z", "x", "e", "q", "s", "t", "m"],
    ]
    """
    simpDat = [
        ["a", "b", "d"],
        ["b", "c", "d"],
        ["a", "b", "e"],
        ["a", "b", "c"],
        ["d", "b", "c"],
    ]

    return simpDat


# 将数据集转化为目标格式
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        if frozenset(trans) in retDict.keys():
            retDict[frozenset(trans)] += 1
        else:
            retDict[frozenset(trans)] = 1
    return retDict


def trans(myItems):
    freqItems = []
    for item in myItems:
        fset = set()
        fcount = 99999
        for i in item:
            fset.add(i[0])
            fcount = min(fcount, i[1])
        freqItems.append((fset, fcount))
    return freqItems


def fp_tree(minSup, mySet):
    initSet = createInitSet(mySet)
    myFPtree, myHeaderTab = createTree(initSet, minSup)  # 形成FP树#
    freqItems = []  # 用于存储频繁项集
    mineTree(myFPtree, myHeaderTab, minSup, set([]), freqItems)  # 获取频繁项集
    return trans(freqItems)

# lie=[0,1,2,3,4,5,6,7,8,9,10,11,18,19,24,25,26,27,28,29,30,31,32,33,34,35]
# tcp=[38,39,40,41,42,43,44,45,50,51]
# udp=[40,41]
# zero_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

input_json = "./features_value_features.json"

if __name__ == "__main__":
    wide = 1#int(sys.argv[1])
    frequent_value = 20#int(sys.argv[2])
    csv_dir_output = "./frequency_output_" + str(frequent_value)
    if not os.path.exists(csv_dir_output):
        os.mkdir(csv_dir_output)

    freqres = {}

    original_vectors = None
    with open(input_json) as fp:
        original_vectors = json.load(fp)

    for class_name, class_vectors in original_vectors.items():
        print(class_name + " start")
        #二维数组
        print("create fptree..")
        initSet = createInitSet(class_vectors)  # 转化为符合格式的事务集
        myFPtree, myHeaderTab = createTree(initSet, frequent_value)  # 形成FP树#
        # myFPtree.disp()  # 打印树
        print("mine fptree..")
        freqItems = []  # 用于存储频繁项集
        mineTree(myFPtree, myHeaderTab, frequent_value, set([]), freqItems)  # 获取频繁项集
        frequent_items = [(list(k),v) for (k,v) in trans(freqItems)]

        data_json = json.dumps(frequent_items, cls=NpEncoder)
        fileObject = open(os.path.join(csv_dir_output, class_name+'.json'), 'w')
        fileObject.write(data_json)
        fileObject.close()
        print(class_name +"保存完毕！")
        freqres[class_name] = frequent_items
        del initSet, myFPtree, myHeaderTab, freqItems, frequent_items
        gc.collect()
    
    data_json = json.dumps(freqres, cls=NpEncoder)
    fileObject = open('./all-freqres-'+str(wide)+'-'+str(frequent_value)+'.json', 'w')
    fileObject.write(data_json)
    fileObject.close()
    print("allres保存完毕！")