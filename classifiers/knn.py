from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
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
        
        #create a dataframe with all training data except the target column
        #X = df.drop(df.iloc[: , :-1], axis=1) # Drop last column of the dataframe
        #y = df.iloc[: , :-1] # target values - get last column of the dataframe
        X = df.drop(df.iloc[: , -1:], axis=1)
        y = df.iloc[:, -1:]
    
    elif dataset == "voting":
        #create a dataframe with all training data except the target column
        X = df.drop(["class"],axis=1)
        y = df["class"]
    
    else:
        #create a dataframe with all training data except the target column
        X = df.drop(df.iloc[: , -1:], axis=1)
        y = df.iloc[:, -1:]

    print(" | Splitting", end = "")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) # Split dataset into random train (80%) and test (20%) subsets:

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

    # Print results: 
    #print(confusion_matrix(y_test, predictions))
    #print(classification_report(y_test, predictions))

    #return confusion_matrix(y_test, predictions)

    return classification_report(y_test, predictions)
    #return classification_report(y_test, predictions)
