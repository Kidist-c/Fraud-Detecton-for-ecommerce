import pandas as pd
class IPToCountryMapper:
    """
    A class to handle IP address preprocessing:
       - convert IpS from float to int 
       -Map Ips to countries using an IP range table
    """
    def __init__(self,ip_country_df:pd.DataFrame):
        """
        parameters
        ----------
        ip_country_df:pd.Dataframe
               Must have columns:
                 -'lower_bound_ip_address'
                 -'upper_bound_ip_address'
                 -'country'
        """
        self.ip_country_df=ip_country_df.copy()
        self._prepare_ip_ranges()
    def _prepare_ip_ranges(self):
        """Ensure IP range columns are integers"""
        self.ip_country_df["lower_bound_ip_address"]=(
            self.ip_country_df["lower_bound_ip_address"].astype(int)
        )
        self.ip_country_df['upper_bound_ip_address']=(
            self.ip_country_df['upper_bound_ip_address'].astype(int)
        )
    @staticmethod
    def convert_ip_to_int(df:pd.DataFrame,ip_column:str='ip_address')->pd.DataFrame:
        """
        Convert IP column from float to int
        """
        df=df.copy()
        df[ip_column]=df[ip_column].astype(int)
        return df
    def map_ip_to_country(self,df:pd.DataFrame,ip_column:str='ip_address',country_column:str='country'):
        """
        Map Ip address in df to countries using th Ip range table
        Returns 
        --------
        Pd.Dataframe with new country column 
        """
        df.copy()
        def lookup_country(ip:int)->str:
            match=self.ip_country_df[(self.ip_country_df['lower_bound_ip_address']<=ip)
                                     & (self.ip_country_df['upper_bound_ip_address']>=ip)]
            return match.iloc[0]['country'] if not match.empty else 'unknown'
        df[country_column]=df[ip_column].apply(lookup_country)
        return df

        