import pandas as pd

DATASET_FOLDER = "/data/datasets"
TEST_FOLDER = "/data/tests"
PREDICTION_FOLDER = "/data/predictions"
IMAGE_FOLDER = "/data/images"

CLASSIFICATION_DTS = [
    "heart",
    "voting",
]

REGRESSION_DTS = [
    "concrete",
    "abalone",
]

# name of the files before being processed
DATASETS = [
    # classification using gnb
    "heart",
    "voting",

    # regression using linear regression
    "concrete",
    "abalone",
]

TEST_SIZE = 0.2 # from 0 to 1. for the datasets that have no test database

reps_votes = []
dems_votes = []

# saves a dataframe into a file
def saveToFile(df, fileName, keyword):
    df.to_csv(f"{fileName[:-4]}{keyword}" , sep=',', index = False, encoding='mbcs')
