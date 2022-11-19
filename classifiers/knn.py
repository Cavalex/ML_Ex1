from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd 

from config import *

# this will ignore some warning about having some labels that were not classified in the test set
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
warnings.filterwarnings('ignore')

# K-Nearest Neighbour 
def knn(df, dataset):
    # Use head() function to return the first 5 rows: 
    df.head() 

    # declaring these variables so that we can use them inside outside the if statement
    X, y, X_train, X_test, y_train, y_test = None, None, None, None, None, None

    print(" | Dropping", end = "")
    if dataset == "amz": # in the amazon dt we want to predict the reviewer, so we get the last column

        global reviewers
        
        #create a dataframe with all training data except the target column
        #X = df.drop(df.iloc[: , :-1], axis=1) # Drop last column of the dataframe
        #y = df.iloc[: , :-1] # target values - get last column of the dataframe
        #X = df.drop(df.iloc[: , -1:], axis=1)
        #y = df.iloc[:, -1:]
        X = df.drop(["Class"],axis=1)
        y = df["Class"]

        # load test file
        fileToBeRead = f".{TEST_FOLDER}/{dataset}_parsed.csv"
        df_test = pd.read_csv(fileToBeRead,  sep=',', on_bad_lines="skip") # reads the csv file and creates a dataframe based on it
        #create a dataframe with all test data except the target column
        X_test = df_test#.drop(df.iloc[: , -1:], axis=1)

        print(" | Classifying", end="")
        classifier = KNeighborsClassifier(n_neighbors=K_NEIGHBOURS_AMAZON, weights="distance") # Use the KNN classifier to fit data:
        classifier.fit(X, y)

        print(" | Predicting")
        predictions = classifier.predict(X_test) # Predict y data with classifier: 

        #print(predictions)

        # save to predictions file
        revs = []
        for i in predictions:
            for k, v in reviewers.items():
                if v == i:
                    revs.append(k)
                    break
        d = {"ID": df_test["ID"], "Class": revs}
        df_save = pd.DataFrame(data=d)
        fileName = f".{PREDICTION_FOLDER}/{dataset}_knn.csv"
        saveToFile(df_save, fileName, ".csv")

        # after saving the results, let's see the accuracy of the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
        classifier.fit(X_train, y_train) 
        predictions = classifier.predict(X_test) # Predict y data with classifier: 

        """ fileToBeRead = f".{IMAGE_FOLDER}/{dataset}_knn.png"
        ConfusionMatrixDisplay.from_predictions(y_test, predictions)
        plt.show()
        plt.savefig(f"{fileToBeRead}") """

        return classification_report(y_test, predictions)
    
    elif dataset == "voting":
        #create a dataframe with all training data except the target column
        X = df.drop(["class"],axis=1)
        y = df["class"]

        # load test file
        fileToBeRead = f".{TEST_FOLDER}/{dataset}_parsed.csv"
        df_test = pd.read_csv(fileToBeRead,  sep=',', on_bad_lines="skip") # reads the csv file and creates a dataframe based on it
        #create a dataframe with all test data except the target column
        X_test = df_test.drop(["ID"],axis=1)


        print(" | Classifying", end="")
        classifier = KNeighborsClassifier(n_neighbors=K_NEIGHBOURS) # Use the KNN classifier to fit data:
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
        fileName = f".{PREDICTION_FOLDER}/{dataset}_knn.csv"
        saveToFile(df_save, fileName, ".csv")

        # after saving the results, let's see the accuracy of the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
        classifier.fit(X_train, y_train) 
        predictions = classifier.predict(X_test) # Predict y data with classifier: 

        fileToBeRead = f".{IMAGE_FOLDER}/{dataset}_knn.png"
        ConfusionMatrixDisplay.from_predictions(y_test, predictions)
        plt.show()
        plt.savefig(f"{fileToBeRead}")

        return classification_report(y_test, predictions)
    
    else:
        #create a dataframe with all training data except the target column
        X = df.drop(df.iloc[: , -1:], axis=1)
        y = df.iloc[:, -1:]

        print(" | Splitting", end = "") # we will split because we have no test file
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE) # Split dataset into random train (80%) and test (20%) subsets:

        #print(" | Fitting", end="")
        # Standardize features by removing mean and scaling to unit variance:
        #scaler = StandardScaler()
        #scaler.fit(X_train)

        #print(" | Transforming", end="")
        #X_train = scaler.transform(X_train)
        #X_test = scaler.transform(X_test)

        print(" | Classifying", end="")
        classifier = KNeighborsClassifier(n_neighbors=K_NEIGHBOURS) # Use the KNN classifier to fit data:
        classifier.fit(X_train, y_train) 

        print(" | Predicting")
        predictions = classifier.predict(X_test) # Predict y data with classifier: 

        fileToBeRead = f".{IMAGE_FOLDER}/{dataset}_knn.png"
        ConfusionMatrixDisplay.from_predictions(y_test, predictions)
        plt.show()
        plt.savefig(f"{fileToBeRead}")

        return classification_report(y_test, predictions)
