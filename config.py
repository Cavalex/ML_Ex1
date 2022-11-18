DATASET_FOLDER = "/data/datasets"
TEST_FOLDER = "/data/tests"
PREDICTION_FOLDER = "/data/predictions"

# name of the files before being processed
DATASETS = [
    "amz",
    "football",
    "voting",
    "heart"
]

TEST_SIZE = 0.2 # from 0 to 1. for the datasets that have no test database

K_NEIGHBOURS = 7 # to be used in the knn algo
DT_DEPTH = 7 # to be used in the dt algo
