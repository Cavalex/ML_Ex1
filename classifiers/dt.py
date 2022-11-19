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
        X = df.drop(["Class"],axis=1)
        y = df["Class"]

        # load test file
        fileToBeRead = f".{TEST_FOLDER}/{dataset}_parsed.csv"
        df_test = pd.read_csv(fileToBeRead,  sep=',', on_bad_lines="skip") # reads the csv file and creates a dataframe based on it
        #create a dataframe with all test data except the target column
        X_test = df_test#.drop(df.iloc[: , -1:], axis=1)

        print(" | Classifying", end="")
        classifier = DecisionTreeClassifier(criterion='entropy', max_depth=DT_DEPTH)
        classifier.fit(X, y)

        print(" | Predicting")
        predictions = classifier.predict(X_test) # Predict y data with classifier: 

        # save to predictions file
        revs = []
        for i in predictions:
            for k, v in reviewers.items():
                if v == i:
                    revs.append(k)
                    break
        d = {"ID": df_test["ID"], "Class": revs}
        df_save = pd.DataFrame(data=d)
        fileName = f".{PREDICTION_FOLDER}/{dataset}_dt.csv"
        saveToFile(df_save, fileName, ".csv")

        # after saving the results, let's see the accuracy of the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
        classifier.fit(X_train, y_train) 
        predictions = classifier.predict(X_test) # Predict y data with classifier: 
        return classification_report(y_test, predictions)

    elif dataset == "voting":
        # create a dataframe with all training data except the target column
        X = df.drop(["class"], axis=1)
        y = df["class"]

        # load test file
        fileToBeRead = f".{TEST_FOLDER}/{dataset}_parsed.csv"
        df_test = pd.read_csv(fileToBeRead,  sep=',', on_bad_lines="skip") # reads the csv file and creates a dataframe based on it
        #create a dataframe with all test data except the target column
        X_test = df_test.drop(["ID"],axis=1)


        print(" | Classifying", end="")
        classifier = DecisionTreeClassifier(criterion='entropy', max_depth=DT_DEPTH)
        classifier.fit(X, y) 

        print(" | Predicting")
        predictions = classifier.predict(X_test) # Predict y data with classifier: 

        # save to predictions file
        pols = []
        for i in predictions:
            if i == 0:
                pols.append("republican")
            else:
                pols.append("democrat")
        d = {"ID": df_test["ID"], "Class": pols}
        #print(d)
        df_save = pd.DataFrame(data=d)
        fileName = f".{PREDICTION_FOLDER}/{dataset}_dt.csv"
        saveToFile(df_save, fileName, ".csv")

        # after saving the results, let's see the accuracy of the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
        classifier.fit(X_train, y_train) 
        predictions = classifier.predict(X_test) # Predict y data with classifier: 
        return classification_report(y_test, predictions)

    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1:]

        print(" | Splitting", end="")

        # Split dataset into random train and test subsets:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

        print(" | Classifying", end="")
        # Train the decision tree classifier by fitting the DecisionTreeClassifier 
        # change max_depth to change proabibilities
        # classifier = DecisionTreeClassifier(max_depth=4)
        classifier = DecisionTreeClassifier(criterion='entropy', max_depth=DT_DEPTH)
        classifier.fit(X_train, y_train)
        #classifier = classifier.fit(X_train, y_train)

        print(" | Predicting")

        predictions = classifier.predict(X_test)
        #print(predictions) 

        # classifier.predict_proba(X_test)
        accuracy_score(y_test, predictions)
        precision_score(y_test, predictions, average='micro')

        # Feature Importance for the results
        #feature_names = X.columns
        #feature_importance = pd.DataFrame(classifier.feature_importances_, index=feature_names).sort_values(0, ascending=False)
        #print(" Feature Importance: ")
        #print(feature_importance)

        return classification_report(y_test, predictions)

