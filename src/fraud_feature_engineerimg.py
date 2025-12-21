import pandas as pd


class TimeBehaviorFeatures:
    """
    Create time-based and transaction
    volocity features for fraud detection
    """
    def __init__(self,user_id_col:str='user_id',
                 purchase_time_col:str='purchase_time',signup_time_col:str='signup_time'):
        self.user_id_col=user_id_col
        self.purchase_time_col=purchase_time_col
        self.signup_time_col=signup_time_col
    def create_time_features(self,df:pd.DataFrame)->pd.dataFrame:
        """
        create basic time-based features 
        from purchase time and signup_time
        """
        df=df.copy()
        df['hour_of_day']=df[self.purchase_time_col].dt.hour
        df['day_of_week']=df[self.purchase_time_col].dt.dayofweek
        df['time_since_signup']=(df[self.purchase_time_col]-df[self.signup_time_col]).dt.total_seconds()
        return df
    def transaction_volocity(self,df:pd.DataFrame)->pd.DataFrame:
        """
        compute number of transaction per user
        """
        df=df.copy()
        df['user_transaction_count']=(
            df.groupby(self.user_id_col)[self.purchase_time_col].transform('count')
        )
        return df
    def apply_all(self,df:pd.DataFrame)->pd.DataFrame:
        """
        Apply all time and behavior features
        """
        df=self.create_time_features(df)
        df=self.transaction_volocity(df)