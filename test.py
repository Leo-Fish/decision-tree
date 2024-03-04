import numpy as np
import pandas as pd
from decision_tree import DecisionTree, dt_train_test
from sklearn import datasets
from sklearn.model_selection import train_test_split

def main():
    iris_data=datasets.load_iris()
    default_md=5
    default_mls=5
    xTrain, xTest, yTrain, yTest = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=42)

    dt1 = DecisionTree('gini', default_md, default_mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    
    dt2 = DecisionTree('entropy', default_md, default_mls)
    trainAcc, testAcc = dt_train_test(dt2, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)

    # dt1.visualize_tree().render('output1', view=True)
    # dt2.visualize_tree().render('output2', view=True)
    
if __name__ == "__main__":
    main()
