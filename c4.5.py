"""
Created on Fri March 20
@author: nichanghao
"""
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import json
from math import log
import operator
import pickle
from plot import createPlot


## 创建数据集
def createDataSet(location):
    IniData = pd.read_table(location, sep='\t', header=None)  # 由于没有列名，读入普通文本，使用read_table
    DNAData = []  # 用来存储DNA数据
    for i in range(IniData.shape[0]):
        s = IniData[0][i].split(" ")
        TempData = []
        for j in range(len(s) // 3):  # 当range(0)时，不执行for循环
            TempData.append(s[3 * j] + s[3 * j + 1] + s[3 * j + 2])  # 一般用加和乘处理数据
        TempData.append(s[-1][0])
        DNAData.append(TempData)

    del TempData
    TrainData = pd.DataFrame(DNAData)
    TrainData.columns = np.arange(0, TrainData.shape[1])
    TrainData.rename(columns={(TrainData.shape[1] - 1): 'classes'}, inplace=True)
    print('划分后的数据属性个数：', TrainData.shape[1] - 1)
    sublist = []
    for i in range(0, TrainData.shape[0]):
        newlist = []  # i的取值是0到60
        for j in range(0, TrainData.shape[1] - 1):  # j的取值时是0-1999
            newlist.append(TrainData[j][i])
        newlist.append(TrainData["classes"][i])
        sublist.append(newlist)
        del newlist
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
              "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
              "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
              "31", "32", "33", "34", "35", "36", "37", "38", "39", "40",
              "41", "42", "43", "44", "45", "46", "47", "48", "49", "50",
              "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "classes"]
    return sublist, labels


## 分割数据集
def splitDataSet(dataset, axis, value):
    """

    :param dataset: 数据集
    :param axis: 划分数据集的特征维度
    :param value: 特征的值
    :return: 符合特征的所有实例（并且自动移除掉这维特征）
    """
    # 循环遍历dataset中每一行的数据
    reDataSet = []
    for featVec in dataset:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis + 1:])
            reDataSet.append(reduceFeatVec)
    return reDataSet


## 计算信息熵
def calcShannonEnt(dataset):
    numEntries = len(dataset)  # 实列的个数
    # print('numEntries',numEntries)
    labelCounts = {}  # 分类标签统计字典，用来统计每个分类标签的概率
    for featVec in dataset:
        currentLabel = featVec[-1]
        # 当前标签不在labelcounts map中，就加入
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


"""
#计算条件熵
def calcConditionEntropy(dataset,i,featList,uniquVals):


    # :param dataset: 数据集 
    # :param i: 维度i
    # :param featList: 数据集特征列表 
    # :param uniquVals: 数据集特征集合
    # :return: 条件熵

    ce = 0.0
    for value in uniquVals:
        subDataSet = splitDataSet(dataset,i,value)
        prob = len(subDataSet)/float(len(dataset))  #极大似然估计概率
        ce += prob * calcShannonEnt(subDataSet)
    return ce

##计算信息增益
def calcInformationGain(dataset,baseEntropy,i):
    featList = [example[i] for example in dataset]
    uniqueVals = set(featList)
    newEntropy = calcConditionEntropy(dataset,i,featList,uniqueVals)
    infoGain = baseEntropy - newEntropy
    return infoGain

"""


# 算法框架
def chooseBestFeatureToSplitByID3(dataset):
    numFeatures = len(dataset[0]) - 1
    # print(len(dataset))
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        newEntropy = 0.0
        splitInfo = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataset, i, value)
            # print('sub_data_set:', len(subDataSet))
            prob = len(subDataSet) / float(len(dataset))  # 极大似然估计概率
            newEntropy += prob * calcShannonEnt(subDataSet)

            splitInfo += -prob * log(prob,2)
        infoGain = baseEntropy - newEntropy
        if splitInfo == 0:
            continue
        infGainRatio = infoGain / splitInfo

        if (infGainRatio > bestInfoGain):
            bestInfoGain = infGainRatio
            bestFeature = i
    return bestFeature


def dict2list(dic: dict):
    keys = dic.keys()
    values = dic.values()
    lst = [(key, value) for key, value in zip(keys, values)]
    return lst


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(dict2list(classCount), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def createTree(dataset, labels):
    classList = [example[-1] for example in dataset]

    if classList.count(classList[0]) == len(classList):  # 类别完全相同则停止继续划分
        return classList[0]
    if len(dataset[0]) == 1:  # 只剩下一列时，返回最多的值
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplitByID3(dataset)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}

    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset, bestFeat, value), subLabels)
    return myTree


def storeTree(inputtree):
    with open("tree.scv", 'w') as json_file:
        json.dump(inputtree, json_file, ensure_ascii=False, cls=json.JSONEncoder)


def grabTree(filename):
    with open(filename, "r") as json_file:
        dic = json.load(json_file)
    return dic


# 测试算法
def classify(inputTree, featLabels, testVector):
    global classLabel
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVector[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVector)
            else:
                classLabel = secondDict[key]

    return classLabel


def test(mytree, labels, location):
    testDataSet, testlabel = createDataSet(location)
    predict = []

    for index in range(0, len(testDataSet)):
        predictlabel = classify(mytree, testlabel, testDataSet[index])
        predict.append(predictlabel)

    sum = 0
    classList = [example[-1] for example in testDataSet]
    for index in range(0, len(testDataSet)):
        if predict[index] == classList[index]:
            sum += 1
    print("正确个数：", sum)
    print("测试数据的个数：", len(testDataSet))
    print("正确率:", sum / len(testDataSet))


dataset, labels = createDataSet("./dataset/dna.data")
# a = chooseBestFeatureToSplitByID3(dataset)
# print(type(dataset))

labelList = labels[:]
# #
Tree = createTree(dataset, labelList)
# #
storeTree(Tree)
createPlot(Tree)
mytree = grabTree("./tree.scv")

test(mytree, labels, "./dataset/dna.test")



