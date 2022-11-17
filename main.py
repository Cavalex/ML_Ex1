import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from parse import *
from classifiers.knn import *
#from classifiers.dt import *
#
from arff_to_csv import *

def parseDatasets(dataset_dfs):
    for dataset in DATASETS:
        dataset_dfs.append(parse(dataset))

def readParsedDatasets(dataset_dfs):
    for dataset in DATASETS:
        fileToBeRead = f".{DATASET_FOLDER}/{dataset}"
        dataset_dfs.append(pd.read_csv(fileToBeRead,  sep=',', on_bad_lines="skip"))

def classifyDatasets(dataset_dfs, classifiers):
    classifications = {}

    for dataset in DATASETS:
        classifications[dataset] = []

    for i in range(len(DATASETS)):
        # classify using each function:
        for classifier in classifiers:
            if DATASETS[i] in classifications:
                print(f"Classifying {DATASET_FOLDER}/{DATASETS[i]}")
                classifications[DATASETS[i]].append(classifier(dataset_dfs[i]))
    
    return classifications

def main():
    classifications = {} # stores the classifications of each dataset according to a specific classifier
    dataset_dfs = [] # stores the dataframes of each dataset after parsing
    classifiers = [knn]

    convertFiles() # converts from arff to csv
    print("\nConverted arff files")
        
    parseDatasets(dataset_dfs)
    print("\nParsed Datasets")

    classifications = classifyDatasets(dataset_dfs, classifiers)
    print("\nClassified Datasets")

if __name__ == "__main__":
    main()
