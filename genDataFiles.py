import pandas as pd; import numpy as np; import pprint; import math; import random as rd
import time; stTotal = time.time() #Start timing
import pathlib; ROOT = pathlib.Path(__file__).parent.resolve(); ROOT = str(ROOT) + "/" # Make file-getting location agnostic
print() # Clear print buffer




#
# General functions
#

#Stat counter
def updateStats(keyPhrase):
    if keyPhrase not in STATS: STATS[keyPhrase] = 1
    else: STATS[keyPhrase] += 1


def shuffle(dataset):
    return dataset.sample(frac=1).reset_index(drop=True)


# Stratified random sampling for train test split
def splitDataset(dataset, className, trainSplit = 0.7):
    byVal = {}

    trainData = pd.DataFrame()
    testData = pd.DataFrame()

    for val in {*dataset[className]}:
        byVal[val] = dataset.loc[dataset[className] == val]
        byVal[val] = shuffle(byVal[val])

        valLen = len(byVal[val].index)
        trainLen = int(valLen * trainSplit)

        trainDataAdd = byVal[val].iloc[0:trainLen]
        testDataAdd = byVal[val].iloc[trainLen:]

        trainData = pd.concat([trainData, trainDataAdd])
        testData = pd.concat([testData, testDataAdd])

    trainData = shuffle(trainData)
    testData = shuffle(testData)

    trainData = trainData.reset_index()
    testData = testData.reset_index()

    trainData = trainData.drop("index", axis=1)
    testData = testData.drop("index", axis=1)

    return trainData, testData


# Discretize a Pandas series of data
def discretize(values, bins):
    maxVal = values.max()
    minVal = values.min()
    rangeSize = maxVal - minVal
    binSize = int(10*rangeSize/bins)
    values = (10*(values-minVal))//binSize
    values = values.astype(int)
    return values


# Discretize a dataset
def discretizeDataset(dataset, className, bins):
    for col in dataset.columns:
        if col == className: continue
        dataset[col] = discretize(dataset[col], bins)
    return dataset



#
# Classifiers
#



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
    dataset = pd.read_csv(ROOT +"PATH")
    dataset = discretizeDataset(dataset, "Outcome", 3)
    trainData, testData = splitDataset(dataset, "Outcome")

    trainData.to_csv(ROOT +"trainingData.csv", index=False)
    testData.to_csv(ROOT +"testingData.csv", index=False)


    print(f"Total time: {(time.time() - stTotal):.3g}s")
    #print(STATS)



#
# Command line execution start below
#

setGlobals() #Set global variables
if __name__ == "__main__": main() #Run main func