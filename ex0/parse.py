import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from arff_to_csv import *

def saveToFile(df, fileName):
    df.to_csv(fileName + "_parsed.csv" , sep=';', index = False, encoding='mbcs')

def main():

    totalRows = 0
    deletedRows = 0

    convertFiles() # converts from arff to csv

    #fileName = "football.csv"
    fileName = "gambling.csv"
    df = pd.read_csv(fileName,  sep=',', on_bad_lines="skip") # reads the csv file and creates a dataframe based on it
    
    gamblingColumns = ["Id","GameID","Username","Bet","CashedOut","Bonus","Profit","BustedAt","PlayDate"]
    footballColumns = ["id_match", "country_id", "country_name", "season", "date_match", "ATT_ATT_diff", "ATT_CEN_diff", "ATT_DEF_diff", "ATT_GOK_diff", "DEF_ATT_diff", "DEF_CEN_diff", "DEF_DEF_diff", "DEF_GOK_diff", "GOK_GOK_diff", "MEN_ATT_diff", "MEN_CEN_diff", "MEN_DEF_diff", "MEN_GOK_diff", "MOV_ATT_diff", "MOV_CEN_diff", "MOV_DEF_diff", "MOV_GOK_diff", "POW_ATT_diff", "POW_CEN_diff", "POW_DEF_diff", "POW_GOK_diff", "SKI_ATT_diff", "SKI_CEN_diff", "SKI_DEF_diff", "SKI_GOK_diff", "outcome"]
    if fileName == "gambling.csv":
        totalRows = len(df.index)
        df = df.loc[df["Bet"] != "?"]
        #df = df.loc[df["Profit"] != "?"]

        df.loc[df['Profit']=='?', "Profit"] = -df['Bet']
        df.loc[df['Bonus']=='?', "Bonus"] = 0

        """ for i in range(len(gamblingColumns)):
            if df.loc[i,"Profit"] == "?":
                df.loc[i, "Profit"] = df.loc[i, "Bet"].astype(float)
        """
        deletedRows  = totalRows - len(df.index)
    elif fileName == "football.csv":
        totalRows = len(df.index)
        for i in range(len(footballColumns)):
            df = df.loc[df[footballColumns[i]] != "?"]
        deletedRows  = totalRows - len(df.index)
        
        # simplify the data
        df = df.replace(['nowin'],0)
        df = df.replace(['win'],1)

        # add the sum columns
        df['ATT_Sum'] = df['ATT_ATT_diff'].astype(float) + df['ATT_DEF_diff'].astype(float) + df['ATT_CEN_diff'].astype(float) + df["ATT_GOK_diff"].astype(float)
    
    saveToFile(df, fileName) # saves to the _parsed file
    #print(df.to_string(index = False))
    print(f"\nInitial total number of rows: {totalRows}")
    if fileName == "gambling.csv":
        print(f"Deleted Rows: {deletedRows}\n")
    elif fileName == "football.csv":
        print(f"Matches deleted: {deletedRows}\n")

    ############################################################################

    if fileName == "gambling.csv":
        df = df.loc[df["Bet"] < 5000] # so we don't get outliers that make visualizing the data too hard

        # needed to make some plots
        df['Bet']=df['Bet'].astype(float)
        df['Profit']=df['Profit'].astype(float)
        #df['Bonus']=df['Bonus'].astype(float)

        #df = df.sample(n=150)
        df.plot(x ='Bet', y='Profit', kind = 'scatter')
        plt.savefig("plot.png")

    elif fileName == "football.csv":
        testValue = "ATT_Sum"
        df[testValue]=df[testValue].astype(float)
        df[testValue].round()
        df[testValue]=df[testValue].astype('int64')
        df['outcome']=df['outcome'].astype(float)

        df = df.sample(n=1000)
        df.plot(x ='outcome', y=testValue, kind = 'scatter')
        
        """ 
        ax = df.plot(x = 'outcome',y=testValue, kind = 'hexbin')
        df.plot(x='outcome', y=testValue, color='Red',ax=ax)
        """
        
        plt.savefig("plot.png")


    """ 
    ‘line’ : line plot (default)
    ‘bar’ : vertical bar plot
    ‘barh’ : horizontal bar plot
    ‘hist’ : histogram
    ‘box’ : boxplot
    ‘kde’ : Kernel Density Estimation plot
    ‘density’ : same as ‘kde’
    ‘area’ : area plot
    ‘pie’ : pie plot
    ‘scatter’ : scatter plot (DataFrame only)
    ‘hexbin’ : hexbin plot (DataFrame only) 
    """


    ############################################################################

if __name__ == "__main__":
    main()
