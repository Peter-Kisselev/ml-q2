import pandas as pd
import numpy as np
import time
import pathlib;
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import utils

pd.set_option("future.no_silent_downcasting", True)

# Hyperparamaters
stTotal = time.time()
ROOT = pathlib.Path(__file__).parent.resolve(); ROOT = str(ROOT) + "/" # Make file-getting location agnostic
le = LabelEncoder()

def chooseAttributes(attributeList: pd.Series) -> pd.Series:
    # filler for now; replacing this is our project
    return attributeList.sample(len(attributeList) // 1.43) # or roughly 70% of the attributes

#Main run
def main():
    trainData, testData = pd.read_csv(str(ROOT)+"/trainingData.csv"), pd.read_csv(str(ROOT)+"/testingData.csv")

    train_numpy = trainData.to_numpy()
    test_numpy = testData.to_numpy()

    train_x, train_y = train_numpy[:, :-1], le.fit_transform(train_numpy[:, -1])
    test_x, test_y = test_numpy[:, :-1], le.fit_transform(test_numpy[:, -1])

    randomForest = RandomForestClassifier(n_estimators = 4)
    randomForest.fit(train_x, train_y)
    y_pred = randomForest.predict(test_x)

    print(f"accuracy: {(y_pred == test_y).sum()}/{len(y_pred)} = {(y_pred == test_y).sum()/len(y_pred)}")
    print(confusion_matrix(test_y, y_pred))

if __name__ == "__main__": main() #Run main func