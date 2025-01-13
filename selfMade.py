import pandas as pd; import numpy as np; import pprint; import math; import random as rd
import time

import test; stTotal = time.time() #Start timing
import pathlib; ROOT = pathlib.Path(__file__).parent.resolve(); ROOT = str(ROOT) + "/" # Make file-getting location agnostic
print() # Clear print buffer
pd.set_option("future.no_silent_downcasting", True)

import matplotlib.pyplot as plt




#
# General functions
#

#Stat counter
def updateStats(keyPhrase):
    if keyPhrase not in STATS: STATS[keyPhrase] = 1
    else: STATS[keyPhrase] += 1


# Generate the confusion matrix
def genConfusionMatrix(testData, className, predictions, classAmnt=2):
    trueVals = testData[className].to_list()
    matrix = [[0 for _ in range(classAmnt)] for _ in range(classAmnt)]
    for i in range(len(predictions)):
        curPred = int(predictions[i])
        curTrue = trueVals[i]
        matrix[curTrue][curPred] += 1
    return matrix


# Cleanly output the confusion matrix
def printConfusionMatrix(matrix):
    sideTxt = "Actual"
    joinStr = "+".join(("-"*5) for _ in range(len(matrix[0])))

    n = 0
    if len(matrix[0]) <= 2:
        print(sideTxt[0] + " Predicted")
    # sideTxt = "tual"
        print(sideTxt[1] + " " + joinStr)
        n = 2
    else:
        print("  Predicted")
        print(sideTxt[0] + " " + joinStr)
        n = 1

    printStr = []
    for ind, row in enumerate(matrix):
        rowStr = []
        for el in row:
            rowStr.append(f" {str(el):3.3}")
        rowStr = " |".join(rowStr)
        printStr.append(rowStr)
        if ind != len(matrix) - 1: printStr.append(joinStr)
    finalStr = []
    for row in printStr:
        if n < len(sideTxt):
            finalStr.append(sideTxt[n] + " " + row)
        else:
            finalStr.append("  " + row)
        n += 1
    printStr = ("\n").join(finalStr)
    print(printStr)
    if n < len(sideTxt):
        print("l " + joinStr)
    else:
        print("  " + joinStr)


# Do confusion matrix calculations and provide key if necessary
def confusionMatrix(testData, className, predictions):
    testData = testData.copy()
    classes = {*testData[className].to_list()}
    classAmnt = len(classes)

    nonInt = any(type(c) != int for c in classes)
    classOrder = []
    classMap = {}

    if nonInt:
        classOrder = sorted([*classes])
        classMap = {c:ind for ind,c in enumerate(classOrder)}
        testData.replace({className: classMap}, inplace=True)
        predictions = [*map(classMap.get, predictions)]

    matrix = genConfusionMatrix(testData, className, predictions, classAmnt)
    print()
    printConfusionMatrix(matrix)
    print()

    if nonInt:
        print(f"Ordering of class values L-R, T->D: {', '.join(classOrder)}")

    print()


# Calculate Euclidean distance
def eucDist(p1, p2):
    return math.sqrt(sum((p1[i]-p2[i])**2 for i in range(len(p1))))


# Find mode of a list/array
def mode(a):
    return max(a, key=lambda x:a.count(x))



#
# Model 1
#
def model1():
    pass


#
# Model 2
#
def model2():
    pass


#
# Evaluate models
#

# Test bayesian model on testing set
def testModel1(bayesTable, instCounter, className, testData):
    correct = 0
    total = 0
    predictions = []
    for ind in range(len(testData.index)):
        instance = testData.iloc[ind]
        predicted = model1(bayesTable, instCounter, instance, className)
        if predicted == instance[className]:
            correct += 1
        total += 1
        predictions.append(predicted)
    accuracy = correct/total
    return accuracy, predictions


# Test bayesian model on testing set
def tesModel2(feature, rules, className, testData):
    correct = 0
    total = 0
    predictions = []
    for ind in range(len(testData.index)):
        instance = testData.iloc[ind]
        predicted = model2(feature, rules, instance)
        if predicted == instance[className]:
            correct += 1
        total += 1
        predictions.append(predicted)
    accuracy = correct/total
    return accuracy, predictions


#
# Top-level execution
#

#Initialize global variables
def setGlobals():
    global STATS
    STATS = {}

    #Reference constants for alphabet and numerals
    global ALPHAREF, NUMREF, CHARORDER
    ALPHAREF = {char for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    NUMREF = {char for char in "0123456789"}
    CHARORDER = [char for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"]


#Main run
def main():
    trainData, testData = pd.read_csv(str(ROOT)+"/trainingData.csv"), pd.read_csv(str(ROOT)+"/testingData.csv")


    print(f"Total time: {(time.time() - stTotal):.3g}s")
    #print(STATS)



#
# Command line execution start below
#

setGlobals() #Set global variables
if __name__ == "__main__": main() #Run main func