import pandas as pd


class CreditCardCleaner:
    """
    Basic Cleaning for credit card fraud dataset
    """
    def clean(self,df:pd.DataFrame)->pd.DataFrame:
        df=df.copy()
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        return df
