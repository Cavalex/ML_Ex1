import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from arff_to_csv import *

def saveToFile(df, fileName):
    df.to_csv(f".{DATASET_FOLDER}/{fileName[:-4]}" + "_parsed.csv" , sep=';', index = False, encoding='mbcs')

def parse(fileName):

    totalRows = 0
    deletedRows = 0

    fileToBeRead = f".{DATASET_FOLDER}/{fileName}"
    df = pd.read_csv(fileToBeRead,  sep=',', on_bad_lines="skip") # reads the csv file and creates a dataframe based on it
    print(f"\nRead {fileToBeRead}, dataframe loaded\n")

    # --------------------------- FOOTBALL ---------------------------

    if fileName == "football.csv":
        totalRows = len(df.index)
        for i in range(len(df.columns)):
            df = df.loc[df[df.columns[i]] != "?"]
        deletedRows  = totalRows - len(df.index)
        
        # simplify the data
        df = df.replace(['nowin'],0)
        df = df.replace(['win'],1)

        # add the sum columns
        #df['ATT_Sum'] = df['ATT_ATT_diff'].astype(float) + df['ATT_DEF_diff'].astype(float) + df['ATT_CEN_diff'].astype(float) + df["ATT_GOK_diff"].astype(float)

    # --------------------------- CONCRETE ---------------------------
    
    # there's really nothing to parse here...
    elif fileName == "concrete.csv":
        pass
        """ totalRows = len(df.index)
        for i in range(len(df.columns)):
            df = df.loc[df[df.columns[i]] != "?"]
        deletedRows  = totalRows - len(df.index) """

    # --------------------------- AMAZON ---------------------------
    
    # we think there's nothing to parse here...
    elif fileName == "amz.csv":
        pass
        """ totalRows = len(df.index)
        for i in range(len(df.columns)):
            df = df.loc[df[df.columns[i]] != "?"]
        deletedRows  = totalRows - len(df.index) """

    # --------------------------- VOTING ---------------------------
    
    elif fileName == "voting.csv":
        totalRows = len(df.index)
        for i in range(len(df.columns)):
            df = df.loc[df[df.columns[i]] != "unknown"]
        deletedRows  = totalRows - len(df.index)

    else:
        print("File not found!")

    saveToFile(df, fileName) # saves to the _parsed file
    #print(df.to_string(index = False))
    print(f"\nParsed file {fileName}")
    print(f"Initial total number of rows: {totalRows}")
    print(f"Deleted Rows: {deletedRows}\n")

    return df
