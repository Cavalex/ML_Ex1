import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy

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

        # replacing the y and n with 1 and 0 respectively
        df = df.replace(['y'], 1)
        df = df.replace(['n'], 0)

        # replacing unknown values with the most probable ones having in account their political parties
        reps = df["class"].value_counts().republican
        dems = df["class"].value_counts().democrat

        new_df = df
        for i in range(len(new_df.columns)):
            new_df = new_df.loc[new_df[new_df.columns[i]] != "unknown"]
        
        # gets the sum of yes'es or no's on each column so that we can calculate a probability of them voting for each problem
        reps_votes = new_df[(new_df["class"] == "republican")].sum().to_numpy()[2:]
        dems_votes = new_df[(new_df["class"] == "democrat")].sum().to_numpy()[2:]

        print(f"reps: {reps} | dems: {dems}")

        #print(reps_votes)
        #print(dems_votes)

        # now we calculate the probablities of voting "y" on a specfic resolution
        reps_votes = [round((reps_votes[i] / reps) * 100) for i in range(len(reps_votes))]
        dems_votes = [round((dems_votes[i] / dems) * 100) for i in range(len(dems_votes))]

        #print(reps_votes)
        #print(dems_votes)


        """ for index in df.index:
            if df.loc[index,"class"] == "republican":
                df.loc[index, "resposta"] = "" """
        
        # we removed all lines with "unknown", it can be used later for classifying data
       """  totalRows = len(df.index)
        for i in range(len(df.columns)):
            df = df.loc[df[df.columns[i]] != "unknown"]
        deletedRows  = totalRows - len(df.index)  """
       



    else:
        print("File not found!")

    saveToFile(df, fileName) # saves to the _parsed file
    #print(df.to_string(index = False))
    print(f"\nParsed file {fileName}")
    print(f"Initial total number of rows: {totalRows}")
    print(f"Deleted Rows: {deletedRows}\n")

    return df
