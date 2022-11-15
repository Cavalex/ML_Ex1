import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from parse import *
from arff_to_csv import *

def main():
    convertFiles() # converts from arff to csv

    print("\nConverted arff files...")
    
    dataset_dfs = []
    for dataset in DATASETS:
        dataset_dfs.append(parse(dataset))

if __name__ == "__main__":
    main()
