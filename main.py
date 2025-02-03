import time; stInit = time.time()
import math
import random
import utils
from typing import Any
import numpy as np
import pandas as pd
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
def mode(arr) -> Any:
    return max(arr, key=lambda x:arr.count(x))


# Compute the euclidean distance from one point to another
def eucDst(p1, p2) -> float:
    return math.sqrt(sum((p2[i]-p1[i])**2 for i in range(len(p1))))


# Use sklearn to classify an element with the decision tree
def classifyDecisionTree(tree, instance):
    return tree.predict([instance])


# Convert
def dfToNumpy(df, className):
    print(df.columns)
    ind = [*df.columns].index(className)
    npDf = df.to_numpy()
    X, y = np.concatenate((npDf[:, :ind], npDf[:, ind+1:]), axis=1), le.fit_transform(npDf[:, ind])
    return X, y


# Normalize a vector with linear sum
def normalizeLin(vec):
    nVal = sum(vec)
    return [el/nVal for el in vec]


#
# Dynamic Semi Random Forest | Core Algorithm
#

# Do whatever idk
def chooseAttributes(attributeList: pd.Series) -> pd.Series:
    # filler for now; replacing this is our project
    return attributeList.sample(len(attributeList) // 1.43) # or roughly 70% of the attributes


# Build the DSRF model
def buildDSRF(data, dataNp, className, attAmnt=5, samples=30, iterations=10):
    dsrf = []
    datalen = len(data.index)

    feats = [*data.columns]
    feats = feats[:feats.index(className)] + feats[feats.index(className)+1:]
    fInds = [ind for ind in range(len(feats))]
    fWeights = [0.001 for _ in range(len(feats))]
    fWeightsCp = fWeights.copy()

    for _ in range(iterations):
        dsrf = []
        for _ in range(samples):
            newDataset = []
            for _ in range(datalen):
                el = data.iloc[random.randint(0, datalen-1)]
                newDataset.append(el)
            newDataset = pd.DataFrame(newDataset)

            choseF = np.random.choice(fInds, size=attAmnt, replace=False, p=normalizeLin(fWeightsCp))
            curFeats = [feats[ind] for ind in choseF]
            curTree = DecisionTreeClassifier()
            curTree.feature_names_in_ = np.array([*curFeats])
            curTree = curTree.fit(dataNp[0], dataNp[1])
            y_pred = curTree.predict(dataNp[0])
            acc = (y_pred == dataNp[1]).sum()/len(y_pred)
            for ind in choseF:
                fWeights[ind] += acc**2
            dsrf.append(curTree)
        fWeightsCp = fWeights.copy()
    print([float(el) for el in fWeights])
    print([float(el) for el in normalizeLin(fWeights)])
    return dsrf


# Predict using the DSRF model
def classifyDSRF(dsrf, instance):
    preds = []
    for tree in dsrf:
        pred = classifyDecisionTree(tree, instance)
        preds.append(pred)
    return mode(preds)



#
# Performance evaluation
#

# Test a singular decision tree
def testDecisionTree(tree, testNp):
    correct = 0
    total = 0
    predictions = []
    for ind in range(len(testNp[0])):
        instNp = testNp[0][ind]
        predicted = classifyDecisionTree(tree, instNp)[0]
        if predicted == testNp[1][ind]:
            correct += 1
        total += 1
        predictions.append(predicted)
    accuracy = correct/total
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

    dsrf = buildDSRF(trainData, train, className, attAmnt=3, samples=20, iterations=100)
    acc, preds = testDSRF(dsrf, test)

    print(f"accuracy: {int(round(acc*len(preds)))}/{len(preds)} = {acc}")
    print(confusion_matrix(test[1], preds))

    # randomForest = RandomForestClassifier(n_estimators = 4)
    # randomForest.fit(train[0], train[1])
    # y_pred = randomForest.predict(test[0])

    # print(f"accuracy: {(y_pred == test[1]).sum()}/{len(y_pred)} = {(y_pred == test[1]).sum()/len(y_pred)}")
    # print(confusion_matrix(test[1], y_pred))


#Main run
def main():
    trainData, testData = pd.read_csv(str(ROOT)+"Data/trainingData.csv"), pd.read_csv(str(ROOT)+"Data/testingData.csv")

    className = "class"
    stInit = time.time()
    doDSRF(trainData, testData, className)
    print(f"\nTotal runtime: {time.time()-stInit:.6f}s")




if __name__ == "__main__": main() #Run main func