import time; stInit = time.time()
import math
import random
import utils
import numpy as np
import pandas as pd
from functools import cmp_to_key
import pathlib; ROOT = pathlib.Path(__file__).parent.resolve(); ROOT = str(ROOT) + "/" # Make file-getting location agnostic
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder; le = LabelEncoder()

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Options and initialization
pd.set_option("future.no_silent_downcasting", True)




#
# Utility functions
#

# Find the mode
def mode(arr):
    return max(arr, key=lambda x:arr.count(x))


# Compute the euclidean distance from one point to another
def eucDst(p1, p2) -> float:
    return math.sqrt(sum((p2[i]-p1[i])**2 for i in range(len(p1))))


# Use sklearn to classify an element with the decision tree
def classifyDecisionTree(tree, instance):
    return tree.predict(instance)


# Convert
def dfToNumpy(df, className):
    # print(df.columns)
    ind = [*df.columns].index(className)
    npDf = df.to_numpy()
    X, y = np.concatenate((npDf[:, :ind], npDf[:, ind+1:]), axis=1), le.fit_transform(npDf[:, ind])
    return X, y


# Normalize a vector with linear sum
def normalizeLin(vec):
    nVal = sum(vec)
    return [el/nVal for el in vec]


# Prepare the probability distribution
def prepDstr(dstr, hits):
    s = sum(dstr)
    avg = s/len(dstr)
    s = 0
    for ind, el in enumerate(dstr):
        if el == 0.001: dstr[ind] = avg
        s += dstr[ind]/hits[ind]
        dstr[ind] = dstr[ind]/hits[ind]
    return np.array([el/(s) for ind, el in enumerate(dstr)])

def weight_sort(x, y): # for an enumerated list (index, accuracy)
    return x[1] < y[1]

#
# Dynamic Semi Random Forest | Core Algorithm
#

# Do whatever idk
def chooseAttributes(attributeList: pd.Series) -> pd.Series:
    # filler for now; replacing this is our project
    return attributeList.sample(len(attributeList) // 1.43) # or roughly 70% of the attributes


# Model Accuracy Function
def accFun(acc):
    return acc ** 5

# Predict using the DSRF model
def classifyDSRF(dsrf, instance):
    preds = []
    for tree in dsrf:
        pred = classifyDecisionTree(tree, [instance])
        preds.append(pred)
    return mode(preds)


# Build the DSRF model
def buildDSRF(data, dataNp, className, testDataset, attAmnt=5, samples=30, batchSize = 10, batchAtts = 5, iterations=10):
    dsrf = []
    datalen = len(data.index)

    feats = [*data.columns]
    featLen = len(feats)
    feats = feats[:feats.index(className)] + feats[feats.index(className)+1:]
    fInds = [ind for ind in range(len(feats))]
    fWeights = np.array([0.001 for _ in range(len(feats))])
    fHits = [1 for _ in range(featLen)]
    fWeightsCp = prepDstr(fWeights.copy(), fHits)

    for i in range(iterations+1):
        print(f"Iteration {i}")
        dsrf = []

        treeAmnt = batchSize if i != iterations else samples
        for _ in range(treeAmnt):
            featAmnt = batchAtts if i != iterations else attAmnt

            newDataInds = np.random.choice(range(datalen), size=datalen, replace=True)
            dataNpNew = (dataNp[0][newDataInds], dataNp[1][newDataInds])

            choseF = np.random.choice(fInds, size=featAmnt, replace=False, p=fWeightsCp)
            curFeats = [feats[ind] for ind in choseF]
            curTree = DecisionTreeClassifier()
            curTree.feature_names_in_ = curFeats
            curTree = curTree.fit(dataNpNew[0], dataNpNew[1])
            y_pred = curTree.predict(dataNpNew[0])
            acc = (y_pred == dataNpNew[1]).sum()/len(y_pred)
            for ind in choseF:
                fWeights[ind] += accFun(acc)
                fHits[ind] += 1

            dsrf.append(curTree)
        fWeightsCp = prepDstr(fWeights.copy(), fHits)
        acc, preds = testDSRF(dsrf, testDataset)
        print(f"accuracy: {int(round(acc*len(preds)))}/{len(preds)} = {acc}")
        print(confusion_matrix(testDataset[1], preds))
    print(f"Number of unhit attributes: {[float(el) for el in fWeights].count(0.001)}")
    # print([float(el) for el in normalizeLin(fWeights)])
    return dsrf


#
# Performance evaluation
#

# Test a singular decision tree
def testDecisionTree2(tree, testNp):
    correct = 0
    total = 0
    predictions = []
    for ind in range(len(testNp[0])):
        instNp = testNp[0][ind]
        predicted = classifyDecisionTree(tree, [instNp])[0]
        if predicted == testNp[1][ind]:
            correct += 1
        total += 1
        predictions.append(predicted)
    accuracy = correct/total
    return accuracy, predictions

def testDecisionTree(tree, testNp):
    predictions = classifyDecisionTree(tree, testNp[0])
    accuracy = (testNp[1] == predictions)/len(predictions)
    return accuracy, predictions


# Test the DSRF
def testDSRF(dsrf, testNp):
    correct = 0
    total = 0
    predictions = []
    for ind in range(len(testNp[0])):
        instNp = testNp[0][ind]
        predicted = classifyDSRF(dsrf, instNp)[0]
        if predicted == testNp[1][ind]:
            correct += 1
        total += 1
        predictions.append(predicted)
    accuracy = correct/total
    return accuracy, predictions


#
# Top-level execution
#

# Main run for Dynamic Semi Random Forest algorithm
def doDSRF(trainData, testData, className):
    train = dfToNumpy(trainData, className)
    test = dfToNumpy(testData, className)

    sampleAmnt = 40

    dsrf = buildDSRF(trainData, train, className, test, attAmnt=45, samples=sampleAmnt, batchSize=30, batchAtts=15, iterations=15)
    acc, preds = testDSRF(dsrf, test)

    print(f"accuracy: {int(round(acc*len(preds)))}/{len(preds)} = {acc}")
    print(confusion_matrix(test[1], preds))

    randomForest = RandomForestClassifier(n_estimators = sampleAmnt)
    randomForest.fit(train[0], train[1])
    y_pred = randomForest.predict(test[0])

    print(f"accuracy: {(y_pred == test[1]).sum()}/{len(y_pred)} = {(y_pred == test[1]).sum()/len(y_pred)}")
    print(confusion_matrix(test[1], y_pred))


#Main run
def main():
    trainData, testData = pd.read_csv(str(ROOT)+"Data/trainingData.csv"), pd.read_csv(str(ROOT)+"Data/testingData.csv")

    className = "class"
    stInit = time.time()
    doDSRF(trainData, testData, className)
    print(f"\nTotal runtime: {time.time()-stInit:.6f}s")




if __name__ == "__main__": main() #Run main func