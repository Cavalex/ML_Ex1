import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from preprocessing import *
from nb import *
from arff_to_csv import *

# remove the outliers from the df
def removeOutliers(df):
    df_summary = df.describe()

    # remove the outliers from the df
    for col in df:
        q1 = df_summary[col].loc["25%"]
        q3 = df_summary[col].loc["75%"]
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        #df.loc[df[col] < lower_bound, col] = df_summary[col].loc["mean"]
        #df.loc[df[col] > upper_bound, col] = df_summary[col].loc["mean"]

        df.drop(df.loc[df[col] < lower_bound, col].index, inplace=True)
        df.drop(df.loc[df[col] > upper_bound, col].index, inplace=True)

        """ try:
            df = df.drop(df.loc[df[col] < lower_bound, col].index, inplace=True)
            df = df.drop(df.loc[df[col] > upper_bound, col].index, inplace=True)
        except:
            pass """

    return df
    
def parseDataset(dataset):
    fileName = f".{DATASET_FOLDER}/{dataset}.csv"
    #if os.path.isfile(fileName):
    df = preprocess(fileName, dataset)
    df = removeOutliers(df)
    return df
    #else:
    #    print(f"Dataset {dataset} not found!")


def parseDatasets(datasets):
    dataset_dfs = []
    for dataset in datasets:
        fileName = f".{DATASET_FOLDER}/{dataset}.csv"
        if os.path.isfile(fileName):
            df = preprocess(fileName, dataset)
            df = removeOutliers(df)

            dataset_dfs.append(df)
        else:
            print(f"Dataset {dataset} not found!")

    return dataset_dfs
