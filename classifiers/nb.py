import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, precision_score

from config import *

# Gaussian Naive Bayes
def nb(df, dataset):

    # declaring these variables so that we can use them inside outside the if statement
    X, y, X_train, X_test, y_train, y_test = None, None, None, None, None, None

    print(" | Dropping", end="")
    if dataset == "amz":  # in the amazon dt we want to predict the reviewer, so we get the last column
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1:]

    elif dataset == "voting":
        # create a dataframe with all training data except the target column
        X = df.drop(["class"], axis=1)
        y = df["class"]

    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1:]

    print(" | Splitting", end="")

    #y = np.ravel(df.iloc[:, -1:])
    # Split dataset into random train and test subsets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

    print(" | Classifying", end="")

    gnb = GaussianNB()
    predcitions = gnb.fit(X_train, y_train).predict(X_test)

    return classification_report(y_test, predcitions)
