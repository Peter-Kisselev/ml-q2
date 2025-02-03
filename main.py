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
    return tree.predict(instance)


# Convert
def dfToNumpy(df, className):
    print(df.columns)
    ind = [*df.columns].index(className)
    npDf = df.to_numpy()
    X, y = np.concatenate((npDf[:, :ind], npDf[:, ind+1:]), axis=1), le.fit_transform(npDf[:, ind])
    return X, y


#
# Dynamic Semi Random Forest | Core Algorithm
#

# Do whatever idk
def chooseAttributes(attributeList: pd.Series) -> pd.Series:
    # filler for now; replacing this is our project
    return attributeList.sample(len(attributeList) // 1.43) # or roughly 70% of the attributes


# Build the DSRF model
def buildDSRF(data, dataNp, className, attAmnt=5, samples=30):
    # Currently code pasted from Random Forest implementation
    dsrf = []
    datalen = len(data.index)
    for _ in range(samples):
        newDataset = []
        for _ in range(datalen):
            el = data.iloc[random.randint(0, datalen-1)]
            newDataset.append(el)
        newDataset = pd.DataFrame(newDataset)
        feats = {*data.columns} - {className}
        for _ in range(attAmnt):
            lstFeats = [*feats]
            chosen = lstFeats[random.randint(0,len(lstFeats)-1)]
            feats.remove(chosen)
        curTree = DecisionTreeClassifier(n_features_in=np.array([*feats]))
        curTree = curTree.fit(dataNp[0], dataNp[1])
        # curTree = buildDecisionTree(data, data, feats, className)
        dsrf.append(curTree)
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

# Test the DSRF
def testDSRF(dsrf, className, testData):
    correct = 0
    total = 0
    predictions = []
    for ind in range(len(testData.index)):
        instance = testData.iloc[ind]
        predicted = classifyDSRF(dsrf, instance)
        if predicted == instance[className]:
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

    randomForest = RandomForestClassifier(n_estimators = 4)
    randomForest.fit(train[0], train[1])
    y_pred = randomForest.predict(test[0])

    print(f"accuracy: {(y_pred == test[1]).sum()}/{len(y_pred)} = {(y_pred == test[1]).sum()/len(y_pred)}")
    print(confusion_matrix(test[1], y_pred))


#Main run
def main():
    trainData, testData = pd.read_csv(str(ROOT)+"Data/trainingDataEx.csv"), pd.read_csv(str(ROOT)+"Data/testingDataEx.csv")

    className = "Outcome"
    stInit = time.time()
    doDSRF(trainData, testData, className)
    print(f"\nTotal runtime: {time.time()-stInit:.6f}s")




if __name__ == "__main__": main() #Run main func