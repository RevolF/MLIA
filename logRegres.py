# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 08:58:38 2017

@author: thor
"""

# implementation of log regression

import numpy as np

def loadDataSet():
    dataMat=[]
    labelMat=[]
#    note that in this text, each row member is the fraction of feature contirbution to overall output
    fr=open('D:\\thor\\documents\\learning_resource\\MACHINE_LEARNING\\machinelearninginaction-master\\mlia_mst\\Ch05\\testSet.txt','r')
    for line in fr.xreadlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
    
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))
    
def gradAscent(dataMatIn,classLabels):
    dataMatrix=np.mat(dataMatIn)
    labelMat=np.mat(classLabels).transpose()
    m,n=dataMatrix.shape
    alpha=0.001
    maxCycles=500
    weights=np.ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=labelMat-h
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights
    
def stocGradAscent0(dataMatrix,classLabels):
    m,n=dataMatrix.shape
    alpha=0.01
    weights=np.ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return weights
    
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n=dataMatrix.shape
    weights=np.ones(n)
    for j in range(numIter):
        dataIndex=range(m)
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(np.random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
    
def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
        
def colicTest():
    frTrain=open('D:\\thor\\documents\\learning_resource\\MACHINE_LEARNING\\machinelearninginaction-master\\mlia_mst\\Ch05\\horseColicTraining.txt','r')
    frTest=open('D:\\thor\\documents\\learning_resource\\MACHINE_LEARNING\\machinelearninginaction-master\\mlia_mst\\Ch05\\horseColicTest.txt','r')
    trainingSet=[]
    trainingLabels=[]
    for line in frTrain.xreadlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[1]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights=stocGradAscent1(np.array(trainingSet),trainingLabels,500)
    errorCount=0
    numTestVec=0.0
    for line in frTest.xreadlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    print('error rate of this test is: %f' % (errorRate))
    return errorRate
    
def multiTest():
    numTests=10
    errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print('after %d iterations the average error rate is: %f' % (numTests,errorSum/float(numTests)))












