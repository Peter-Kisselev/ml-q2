import pandas as pd
import numpy as np
import math
from typing import Any
import time; stInit = time.time()
import pathlib; ROOT = pathlib.Path(__file__).parent.resolve(); ROOT = str(ROOT) + "/" # Make file-getting location agnostic
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder; le = LabelEncoder()
import utils

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
# Algorithm
#



# Do whatever idk
def chooseAttributes(attributeList: pd.Series) -> pd.Series:
    # filler for now; replacing this is our project
    return attributeList.sample(len(attributeList) // 1.43) # or roughly 70% of the attributes



#
# Top-level execution
#

# Main run for Dynamic Semi Random Forest algorithm
def dsrf(trainData, testData):
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
    trainData, testData = pd.read_csv(str(ROOT)+"/trainingDataEx.csv"), pd.read_csv(str(ROOT)+"/testingDataEx.csv")
    dsrf(trainData, testData)
    print(f"\nTotal runtime: {stInit-time.time():.3f}")




if __name__ == "__main__": main() #Run main func