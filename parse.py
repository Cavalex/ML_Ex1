import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy
import random

from arff_to_csv import *
from config import *


def saveToFile(df, fileName):
    df.to_csv(f".{DATASET_FOLDER}/{fileName}" + "_parsed.csv" , sep=';', index = False, encoding='mbcs')

def parse(fileName):

    totalRows = 0
    deletedRows = 0

    fileToBeRead = f".{DATASET_FOLDER}/{fileName}.csv"
    df = pd.read_csv(fileToBeRead,  sep=',', on_bad_lines="skip") # reads the csv file and creates a dataframe based on it
    #print(f"\nLoaded {fileToBeRead}")

    # --------------------------- FOOTBALL ---------------------------

    if fileName == "football":

        # deleting unnecessary columns
        del df["id_match"]
        del df["country_id"]
        del df["country_name"]
        del df["season"]
        del df["date_match"]

        totalRows = len(df.index)
        for i in range(len(df.columns)):
            df = df.loc[df[df.columns[i]] != "?"]
        deletedRows  = totalRows - len(df.index)
        
        # simplify the data
        df = df.replace(["nowin"],0)
        df = df.replace(["win"],1)

        # add the sum columns
        #df['ATT_Sum'] = df['ATT_ATT_diff'].astype(float) + df['ATT_DEF_diff'].astype(float) + df['ATT_CEN_diff'].astype(float) + df["ATT_GOK_diff"].astype(float)

    # --------------------------- HEART ---------------------------
    
    elif fileName == "heart":

        #df.columns = df.columns.str.lstrip("'")

        # deleting unnecessary columns
        # damn these quotation marks
        del df["\'thal\'"]
        del df["\'slope\'"]
        del df["\'ca\'"]

        # replacing the string values with ints
        df = df.replace({"female": 0, "male": 1})

        df = df.replace({"typ_angina": 1, "atyp_angina": 2, "non_anginal": 3, "asympt": 4})

        df = df.replace({"f": 0, "t": 1})

        df = df.replace({"normal": 0, "st_t_wave_abnormality": 1, "left_vent_hyper": 2})

        df = df.replace({"no": 0, "yes": 1})

        df = df.replace({"\'<50\'": 0, "\'>50_1\'": 1})
    
        totalRows = len(df.index)
        for i in range(len(df.columns)):
            df = df.loc[df[df.columns[i]] != "?"]
        deletedRows  = totalRows - len(df.index)

        # convert everything to int
        for col in df:
            df[col] = df[col].astype('int')

    # --------------------------- AMAZON ---------------------------
    
    # we think there's nothing to parse here
    elif fileName == "amz":
        
        del df["Class"]

        totalRows = len(df.index)
        """ for i in range(len(df.columns)):
            df = df.loc[df[df.columns[i]] != "?"]
        deletedRows  = totalRows - len(df.index) """

    # --------------------------- VOTING ---------------------------
    
    elif fileName == "voting":

        # delete the id column because we don't need it to classify anything
        del df["ID"]

        # replacing the y and n with 1 and 0 respectively
        df = df.replace(["y"], 1)
        df = df.replace(["n"], 0)

        # replacing unknown values with the most probable ones having in account their political parties
        reps = df["class"].value_counts().republican
        dems = df["class"].value_counts().democrat

        # replacing the class names with a bit to make later processing easier
        df = df.replace(["democrat"], 1)
        df = df.replace(["republican"], 0)

        new_df = df
        for i in range(len(new_df.columns)):
            new_df = new_df.loc[new_df[new_df.columns[i]] != "unknown"]
        
        # gets the sum of yes'es or no's on each column so that we can calculate a probability of them voting for each problem
        reps_votes = new_df[(new_df["class"] == 1)].sum().to_numpy()[2:]
        dems_votes = new_df[(new_df["class"] == 0)].sum().to_numpy()[2:]

        #print(f"reps: {reps} | dems: {dems}")

        #print(reps_votes)
        #print(dems_votes)

        # now we calculate the probablities of voting "y" on a specfic resolution
        reps_votes = [round((reps_votes[i] / reps) * 100) for i in range(len(reps_votes))]
        dems_votes = [round((dems_votes[i] / dems) * 100) for i in range(len(dems_votes))]

        #print(reps_votes)
        #print(dems_votes)

        # replace the unkowns in the original df with the respective values according to the probabilities
        for index in df.index:
            if df.loc[index,"class"] == 0: # republican
                for col in df:
                    col_index = 2
                    if df.loc[index, col] == "unknown":
                        df.loc[index, col] = 1 if random.randint(0,100) < reps_votes[col_index] else 0
                    col_index += 1
            else:
                for col in df:
                    col_index = 2
                    if df.loc[index, col] == "unknown":
                        df.loc[index, col] = 1 if random.randint(0,100) < dems_votes[col_index] else 0
                    col_index += 1
        
        # convert everything to int
        for col in df:
            df[col] = df[col].astype('int')

        # we removed all lines with "unknown", it can be used later for classifying data
        totalRows = len(df.index)


    else:
        print("File not found!")

    saveToFile(df, fileName) # saves to the _parsed file
    #print(df.to_string(index = False))
    print(f"Parsed {fileToBeRead}")
    #print(f"Initial total number of rows: {totalRows}")
    #print(f"Deleted Rows: {deletedRows}\n")

    return df
