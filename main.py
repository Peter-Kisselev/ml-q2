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


#
# Dynamic Semi Random Forest | Core Algorithm
#

# Do whatever idk
def chooseAttributes(attributeList: pd.Series) -> pd.Series:
    # filler for now; replacing this is our project
    return attributeList.sample(len(attributeList) // 1.43) # or roughly 70% of the attributes


# Build the DSRF model
def buildDSRF(data, className, attAmnt=5, samples=30):
    # Currently code pasted from Random Forest implementation
    dsrf = []
    datalen = len(data.index)
    for _ in range(samples):
        newDataset = []
        for i in range(datalen):
            el = data.iloc[random.randint(0, datalen-1)]
            newDataset.append(el)
        newDataset = pd.DataFrame(newDataset)
        feats = {*data.columns}
        for i in range(attAmnt):
            lstFeats = [*feats]
            chosen = lstFeats[random.randint(0,len(lstFeats)-1)]
            feats.remove(chosen)
        curTree = DecisionTreeClassifier()
        curTree = curTree.fit(data, className)
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
    train_numpy = trainData.to_numpy()
    test_numpy = testData.to_numpy()

    train_x, train_y = train_numpy[:, :-1], le.fit_transform(train_numpy[:, -1])
    test_x, test_y = test_numpy[:, :-1], le.fit_transform(test_numpy[:, -1])

    randomForest = RandomForestClassifier(n_estimators = 4)
    randomForest.fit(train_x, train_y)
    y_pred = randomForest.predict(test_x)

    print(f"accuracy: {(y_pred == test_y).sum()}/{len(y_pred)} = {(y_pred == test_y).sum()/len(y_pred)}")
    print(confusion_matrix(test_y, y_pred))


#Main run
def main():
    trainData, testData = pd.read_csv(str(ROOT)+"Data/trainingDataEx.csv"), pd.read_csv(str(ROOT)+"Data/testingDataEx.csv")

    className = "class"
    stInit = time.time()
    doDSRF(trainData, testData, className)
    print(f"\nTotal runtime: {stInit-time.time():.3f}")




if __name__ == "__main__": main() #Run main func