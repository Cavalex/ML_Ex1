import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from parse import *
from classifiers.knn import *
from classifiers.dt import *
from classifiers.nb import *

from arff_to_csv import *
from plot import *

#def readParsedDatasets(dataset_dfs):
#    for dataset in DATASETS:
#        fileToBeRead = f".{DATASET_FOLDER}/{dataset}"
#        dataset_dfs.append(pd.read_csv(fileToBeRead,  sep=',', on_bad_lines="skip"))

def plotDatasets():
    for dataset in DATASETS:
        fileToBeRead = f".{DATASET_FOLDER}/{dataset}_parsed.csv"
        if os.path.isfile(fileToBeRead):
            df = pd.read_csv(fileToBeRead,  sep=',', on_bad_lines="skip") # reads the csv file and creates a dataframe based on it
            plot(dataset, df)
        else:
            print(f"Dataset {dataset} not found!")

def parseDatasets():
    dataset_dfs = []
    for dataset in DATASETS:
        fileName = f".{DATASET_FOLDER}/{dataset}.csv"
        if os.path.isfile(fileName):
            dataset_dfs.append(parse(fileName, dataset))
        else:
            print(f"Dataset {dataset} not found!")
    return dataset_dfs


def parseTestDatasets():
    dataset_dfs = []
    for dataset in DATASETS:
        fileName = f".{TEST_FOLDER}/{dataset}.csv"
        if os.path.isfile(fileName):
            dataset_dfs.append(parse(fileName, dataset))
        else:
            print(f"Dataset {dataset} not found!")
    return dataset_dfs


def classifyDatasets(classifiers):
    classifications = {}

    for dataset in DATASETS:
        classifications[dataset] = []

    for dataset in DATASETS:
        fileToBeRead = f".{DATASET_FOLDER}/{dataset}_parsed.csv"
        if os.path.isfile(fileToBeRead):
            df = pd.read_csv(fileToBeRead,  sep=',', on_bad_lines="skip") # reads the csv file and creates a dataframe based on it
            # classify using each function:
            for classifier in classifiers:
                print()
                if dataset in classifications:
                    classifier_name = classifier.__name__
                    print(f"Classifying {fileToBeRead} with {classifier_name}", end="")
                    classifications[dataset].append(classifier(df, dataset))
        else:
            print(f"Dataset {dataset} not found!")

    return classifications


def main():
    classifications = {} # stores the classifications of each dataset according to a specific classifier
    dataset_dfs = [] # stores the dataframes of each dataset after parsing
    test_dfs = [] # stores the dataframes of each test dataset after parsing
    classifiers = [knn, dt, nb] # stores the classifier functions we will be using to train our models

    convertFiles() # converts from arff to csv
    print("\nConverted arff files\n")

    dataset_dfs = parseDatasets()
    print("Parsed Training Datasets\n")

    test_dfs = parseTestDatasets()
    print("Parsed Test Datasets\n")

    classifications = classifyDatasets(classifiers)
    print("Classified Datasets\n")

    plotDatasets()
    print("Plotted Datasets\n")

    # this will print the results of our training
    for dataset in classifications:
        print("\n----------------------------------------------------------------------------")
        print(dataset)
        for i in range(len(classifiers)):
            try:
                print("\n\t" + classifiers[i].__name__ + ":\n")
                print(classifications[dataset][i])
            except:
                print(f"Dataset {dataset} not found!")
        
    #print(classifications)

if __name__ == "__main__":
    main()
