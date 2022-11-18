import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from parse import *
from classifiers.knn import *
from classifiers.dt import *
#
from arff_to_csv import *

def parseDatasets(dataset_dfs):
    for dataset in DATASETS:
        dataset_dfs.append(parse(dataset))

#def readParsedDatasets(dataset_dfs):
#    for dataset in DATASETS:
#        fileToBeRead = f".{DATASET_FOLDER}/{dataset}"
#        dataset_dfs.append(pd.read_csv(fileToBeRead,  sep=',', on_bad_lines="skip"))

def classifyDatasets(classifiers):
    classifications = {}

    for dataset in DATASETS:
        classifications[dataset] = []

    for dataset in DATASETS:
        fileToBeRead = f".{DATASET_FOLDER}/{dataset}_parsed.csv"
        df = pd.read_csv(fileToBeRead,  sep=';', on_bad_lines="skip") # reads the csv file and creates a dataframe based on it
        # classify using each function:
        for classifier in classifiers:
            if dataset in classifications:
                print(f"Classifying {fileToBeRead}", end="")
                classifications[dataset].append(classifier(df, dataset))

    return classifications

def main():
    classifications = {} # stores the classifications of each dataset according to a specific classifier
    dataset_dfs = [] # stores the dataframes of each dataset after parsing
    classifiers = [knn,dt]

    convertFiles() # converts from arff to csv
    print("\nConverted arff files")
        
    parseDatasets(dataset_dfs)
    print("\nParsed Datasets")

    classifications = classifyDatasets(classifiers)
    print("\nClassified Datasets")

    for dataset in classifications:
        print("\n----------------------------------------------------------------------------")
        print(dataset)
        for i in range(len(classifiers)):
            print("\n\t" + classifiers[i].__name__ + ":\n")
            print(classifications[dataset][i])
    #print(classifications)

if __name__ == "__main__":
    main()
