from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd 

# K-Nearest Neighbour 
def knn(df):
    # Use head() function to return the first 5 rows: 
    df.head() 

    #create a dataframe with all training data except the target column
    X = df.drop(df.iloc[: , :-1], axis=1) # Drop last column of the dataframe
    
    #separate target values
    y = df.iloc[: , :-1] # get last column of the dataframe

    print("Splitting...")

    # Split dataset into random train (80%) and test (20%) subsets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    print("Fitting...")

    # Standardize features by removing mean and scaling to unit variance:
    scaler = StandardScaler()
    scaler.fit(X_train)

    print("Transforming...")

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print("Classifying...")

    # Use the KNN classifier to fit data:
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train) 

    print("Predicting...")

    # Predict y data with classifier: 
    y_predict = classifier.predict(X_test)

    # Print results: 
    #print(confusion_matrix(y_test, y_predict))
    #print(classification_report(y_test, y_predict))
