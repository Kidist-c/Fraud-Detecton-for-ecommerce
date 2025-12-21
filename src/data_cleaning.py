# data cleaning function for Fraud Data set
import pandas as pd

def clean_fraud_data(df):
    df=df.copy()
    # convert timestamps
    df["signup_time"]=pd.to_datetime(df["signup_time"])
    df["purchase_time"]=pd.to_datetime(df["purchase_time"])

    # drop the duplicate if there is any
    df.drop_duplicates(inplace=True)
    """ handle misssing Values:
     * NUmerical cols: Fill with Median 
     * Catagorical cols: Fill with Mode

    """
    df["age"].fillna(df["age"].median(),inplace=True)# fill the age column missing Value with median if any
    df["sex"].fillna(df["sex"].mode(),inplace=True) # fill the sex column msiing value with mode if any
    

    return df