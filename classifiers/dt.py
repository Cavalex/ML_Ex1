import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score

from config import *

# Decision Trees
def dt(df, dataset):

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

    # Split dataset into random train and test subsets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    print(" | Classifying", end="")
    # Train the decision tree classifier by fitting the DecisionTreeClassifier 
    # change max_depth to change proabibilities
    classifier = DecisionTreeClassifier(max_depth=4)
    classifier = classifier.fit(X_train, y_train)

    print(" | Predicting", end="")

    predictions = classifier.predict(X_test)
    print(predictions) 

    # classifier.predict_proba(X_test)
    accuracy_score(y_test, predictions)
    precision_score(y_test, predictions, average='micro')

    return classification_report(y_test, predictions)

